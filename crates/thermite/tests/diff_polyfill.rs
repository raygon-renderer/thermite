//! Polyfill-focused differential tests.
//!
//! Operations with no native hardware instruction are emulated by a *polyfill*
//! (`backend/*/polyfills/`). These are the highest-risk code in the library -
//! this audit found **five** production bugs here (all since fixed). Every
//! polyfill-backed register op is checked against an **independent pure-Rust
//! oracle** (not the scalar backend, which may route through the same generic
//! polyfill and hide a shared bug).
//!
//! `mod fixed` contains regression tests for the four defects this file's
//! audit found and that have been fixed (P1–P4); see TESTING.md for the
//! root-cause analysis. All tests here run in the default suite.
//!
//! `X86V2` and `X86V3` share these polyfills, so a defect in one is a defect
//! in both; the regression tests cover both backends.
#![cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32"))]

mod harness;

use harness::Tol;

use thermite::register::{
    BitshiftRegister as _, FloatRegister as _, IntegerRegister as _, Register as _, SignedIntegerRegister as _,
    SignedRegister as _, UnsignedIntegerRegister as _,
};
use thermite::simd::Simd;

// ===========================================================================
// Verified-correct polyfills - always-green.
// ===========================================================================

/// `mullo` (low half of the product) - exercises the now-fixed 64-bit
/// `_mm{,256}_mullo_epi64x` emulation. Correct for every width.
macro_rules! mullo_for {
    ($($reg:ident: $b:ty, $e:ty, $l:expr);* $(;)?) => {
        #[test]
        fn mullo() {
            $( oracle_binary!($l, <$b as Simd>::$reg, $e, mullo, |a, b| a.wrapping_mul(b), Tol::Exact); )*
        }
    };
}

/// Bit population / scan ops that are correct everywhere they're tested here.
macro_rules! popcount_for {
    ($($reg:ident: $b:ty, $e:ty, $l:expr);* $(;)?) => {
        #[test]
        fn popcount() {
            $(
                oracle_unary!($l, <$b as Simd>::$reg, $e, count_ones, |x| x.count_ones() as $e, Tol::Exact);
                oracle_unary!($l, <$b as Simd>::$reg, $e, count_zeros, |x| x.count_zeros() as $e, Tol::Exact);
            )*
        }
    };
}

/// Byte/bit reversal and rotates - correct for every width.
macro_rules! bitperm_for {
    ($($reg:ident: $b:ty, $e:ty, $l:expr);* $(;)?) => {
        #[test]
        fn bitperm() {
            $(
                oracle_unary!($l, <$b as Simd>::$reg, $e, swap_bytes, |x| x.swap_bytes(), Tol::Exact);
                oracle_unary!($l, <$b as Simd>::$reg, $e, reverse_bits, |x| x.reverse_bits(), Tol::Exact);
                oracle_shift!($l, <$b as Simd>::$reg, $e, rol, |x, s| x.rotate_left(s));
                oracle_shift!($l, <$b as Simd>::$reg, $e, ror, |x, s| x.rotate_right(s));
            )*
        }
    };
}

macro_rules! for_lztz {
    ($l:expr, $ut:ty, $e:ty) => {{
        oracle_unary!($l, $ut, $e, leading_zeros, |x| x.leading_zeros() as $e, Tol::Exact);
        oracle_unary!($l, $ut, $e, trailing_zeros, |x| x.trailing_zeros() as $e, Tol::Exact);
    }};
}
macro_rules! for_signed {
    ($l:expr, $ut:ty, $e:ty, $w:ty) => {{
        oracle_shift!($l, $ut, $e, sra, |x, s| x >> s);
        oracle_binary!(
            $l,
            $ut,
            $e,
            avg_floor,
            |a, b| (((a as $w) + (b as $w)) >> 1) as $e,
            Tol::Exact
        );
        oracle_binary!(
            $l,
            $ut,
            $e,
            avg_ceil,
            |a, b| (((a as $w) + (b as $w) + 1) >> 1) as $e,
            Tol::Exact
        );
    }};
}
macro_rules! for_float {
    ($l:expr, $ut:ty, $e:ty) => {{
        oracle_binary!($l, $ut, $e, copysign, |a, b| a.copysign(b), Tol::Exact);
        oracle_unary!($l, $ut, $e, fract, |x| x - x.trunc(), Tol::Exact);
    }};
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
use super::*;
use thermite::backend::x86_v1::X86V1;
use thermite::backend::x86_v2::X86V2;
use thermite::backend::x86_v3::X86V3;

mod correct {
    use super::*;

    mullo_for! {
        i32x4: X86V3, i32, "v3 i32x4"; i32x8: X86V3, i32, "v3 i32x8";
        i64x2: X86V3, i64, "v3 i64x2"; i64x4: X86V3, i64, "v3 i64x4";
        u32x4: X86V3, u32, "v3 u32x4"; u32x8: X86V3, u32, "v3 u32x8";
        u64x2: X86V3, u64, "v3 u64x2"; u64x4: X86V3, u64, "v3 u64x4";
        i32x4: X86V2, i32, "v2 i32x4"; i64x2: X86V2, i64, "v2 i64x2";
        u32x4: X86V2, u32, "v2 u32x4"; u64x2: X86V2, u64, "v2 u64x2";
    }
    popcount_for! {
        i32x4: X86V3, i32, "v3 i32x4"; u32x4: X86V3, u32, "v3 u32x4";
        i64x2: X86V3, i64, "v3 i64x2"; u64x2: X86V3, u64, "v3 u64x2";
        i32x8: X86V3, i32, "v3 i32x8"; u32x8: X86V3, u32, "v3 u32x8";
        i64x4: X86V3, i64, "v3 i64x4"; u64x4: X86V3, u64, "v3 u64x4";
        i32x4: X86V2, i32, "v2 i32x4"; u64x2: X86V2, u64, "v2 u64x2";
    }
    bitperm_for! {
        i32x4: X86V3, i32, "v3 i32x4"; u32x4: X86V3, u32, "v3 u32x4";
        i64x2: X86V3, i64, "v3 i64x2"; u64x2: X86V3, u64, "v3 u64x2";
        i32x8: X86V3, i32, "v3 i32x8"; u32x8: X86V3, u32, "v3 u32x8";
        i64x4: X86V3, i64, "v3 i64x4"; u64x4: X86V3, u64, "v3 u64x4";
        i32x4: X86V2, i32, "v2 i32x4"; i64x2: X86V2, i64, "v2 i64x2";
    }

    // `mulhi` is correct for 64-bit lanes (wrong for 32-bit, see bugs::mulhi32).
    #[test]
    fn mulhi_64() {
        oracle_binary!(
            "v3 i64x2",
            <X86V3 as Simd>::i64x2,
            i64,
            mulhi,
            |a, b| (((a as i128) * (b as i128)) >> 64) as i64,
            Tol::Exact
        );
        oracle_binary!(
            "v3 u64x2",
            <X86V3 as Simd>::u64x2,
            u64,
            mulhi,
            |a, b| (((a as u128) * (b as u128)) >> 64) as u64,
            Tol::Exact
        );
        oracle_binary!(
            "v3 i64x4",
            <X86V3 as Simd>::i64x4,
            i64,
            mulhi,
            |a, b| (((a as i128) * (b as i128)) >> 64) as i64,
            Tol::Exact
        );
        oracle_binary!(
            "v3 u64x4",
            <X86V3 as Simd>::u64x4,
            u64,
            mulhi,
            |a, b| (((a as u128) * (b as u128)) >> 64) as u64,
            Tol::Exact
        );
        oracle_binary!(
            "v2 i64x2",
            <X86V2 as Simd>::i64x2,
            i64,
            mulhi,
            |a, b| (((a as i128) * (b as i128)) >> 64) as i64,
            Tol::Exact
        );
    }

    // saturating arithmetic is correct for the unsigned widths.
    #[test]
    fn saturating_unsigned() {
        oracle_binary!(
            "v3 u32x4",
            <X86V3 as Simd>::u32x4,
            u32,
            saturating_add,
            |a, b| a.saturating_add(b),
            Tol::Exact
        );
        oracle_binary!(
            "v3 u32x4",
            <X86V3 as Simd>::u32x4,
            u32,
            saturating_sub,
            |a, b| a.saturating_sub(b),
            Tol::Exact
        );
        oracle_binary!(
            "v3 u64x2",
            <X86V3 as Simd>::u64x2,
            u64,
            saturating_add,
            |a, b| a.saturating_add(b),
            Tol::Exact
        );
        oracle_binary!(
            "v3 u64x2",
            <X86V3 as Simd>::u64x2,
            u64,
            saturating_sub,
            |a, b| a.saturating_sub(b),
            Tol::Exact
        );
        oracle_binary!(
            "v3 u32x8",
            <X86V3 as Simd>::u32x8,
            u32,
            saturating_add,
            |a, b| a.saturating_add(b),
            Tol::Exact
        );
        oracle_binary!(
            "v3 u32x8",
            <X86V3 as Simd>::u32x8,
            u32,
            saturating_sub,
            |a, b| a.saturating_sub(b),
            Tol::Exact
        );
        oracle_binary!(
            "v3 u64x4",
            <X86V3 as Simd>::u64x4,
            u64,
            saturating_add,
            |a, b| a.saturating_add(b),
            Tol::Exact
        );
        oracle_binary!(
            "v3 u64x4",
            <X86V3 as Simd>::u64x4,
            u64,
            saturating_sub,
            |a, b| a.saturating_sub(b),
            Tol::Exact
        );
    }

    // leading/trailing zeros are correct for everything except u64.
    #[test]
    fn bitscan_ok() {
        for_lztz!("v3 i32x4", <X86V3 as Simd>::i32x4, i32);
        for_lztz!("v3 u32x4", <X86V3 as Simd>::u32x4, u32);
        for_lztz!("v3 i64x2", <X86V3 as Simd>::i64x2, i64);
        for_lztz!("v3 i32x8", <X86V3 as Simd>::i32x8, i32);
        for_lztz!("v3 u32x8", <X86V3 as Simd>::u32x8, u32);
        for_lztz!("v3 i64x4", <X86V3 as Simd>::i64x4, i64);
        for_lztz!("v2 i64x2", <X86V2 as Simd>::i64x2, i64);
    }

    // Signed-only: arithmetic shift right + floor/ceil averages.
    #[test]
    fn signed_extras() {
        for_signed!("v3 i32x4", <X86V3 as Simd>::i32x4, i32, i64);
        for_signed!("v3 i64x2", <X86V3 as Simd>::i64x2, i64, i128);
        for_signed!("v3 i32x8", <X86V3 as Simd>::i32x8, i32, i64);
        for_signed!("v3 i64x4", <X86V3 as Simd>::i64x4, i64, i128);
        for_signed!("v2 i32x4", <X86V2 as Simd>::i32x4, i32, i64);
        for_signed!("v2 i64x2", <X86V2 as Simd>::i64x2, i64, i128);
    }

    // Unsigned PAVG ceiling average.
    #[test]
    fn unsigned_avg() {
        oracle_binary!(
            "v3 u32x4",
            <X86V3 as Simd>::u32x4,
            u32,
            avg,
            |a, b| (((a as u64) + (b as u64) + 1) >> 1) as u32,
            Tol::Exact
        );
        oracle_binary!(
            "v3 u64x2",
            <X86V3 as Simd>::u64x2,
            u64,
            avg,
            |a, b| (((a as u128) + (b as u128) + 1) >> 1) as u64,
            Tol::Exact
        );
        oracle_binary!(
            "v3 u32x8",
            <X86V3 as Simd>::u32x8,
            u32,
            avg,
            |a, b| (((a as u64) + (b as u64) + 1) >> 1) as u32,
            Tol::Exact
        );
        oracle_binary!(
            "v3 u64x4",
            <X86V3 as Simd>::u64x4,
            u64,
            avg,
            |a, b| (((a as u128) + (b as u128) + 1) >> 1) as u64,
            Tol::Exact
        );
    }

    // Float polyfills with bit-exact Rust oracles.
    #[test]
    fn float_ops() {
        for_float!("v3 f32x8", <X86V3 as Simd>::f32x8, f32);
        for_float!("v3 f64x4", <X86V3 as Simd>::f64x4, f64);
        for_float!("v2 f32x4", <X86V2 as Simd>::f32x4, f32);
        for_float!("v2 f64x2", <X86V2 as Simd>::f64x2, f64);
    }
}

// ===========================================================================
// X86V1 (SSE2) - these polyfills are *distinct implementations* from the
// v2/v3 ones (no SSE4.1 blendv, no pshufb, SWAR popcount, magic-number
// rounding), so they get their own full pass against the same Rust oracles.
// ===========================================================================
mod v1 {
    use super::*;

    mullo_for! {
        i32x4: X86V1, i32, "v1 i32x4"; i64x2: X86V1, i64, "v1 i64x2";
        u32x4: X86V1, u32, "v1 u32x4"; u64x2: X86V1, u64, "v1 u64x2";
    }
    popcount_for! {
        i32x4: X86V1, i32, "v1 i32x4"; u32x4: X86V1, u32, "v1 u32x4";
        i64x2: X86V1, i64, "v1 i64x2"; u64x2: X86V1, u64, "v1 u64x2";
    }
    bitperm_for! {
        i32x4: X86V1, i32, "v1 i32x4"; u32x4: X86V1, u32, "v1 u32x4";
        i64x2: X86V1, i64, "v1 i64x2"; u64x2: X86V1, u64, "v1 u64x2";
    }

    #[test]
    fn mulhi() {
        oracle_binary!(
            "v1 i32x4",
            <X86V1 as Simd>::i32x4,
            i32,
            mulhi,
            |a, b| (((a as i64) * (b as i64)) >> 32) as i32,
            Tol::Exact
        );
        oracle_binary!(
            "v1 u32x4",
            <X86V1 as Simd>::u32x4,
            u32,
            mulhi,
            |a, b| (((a as u64) * (b as u64)) >> 32) as u32,
            Tol::Exact
        );
        oracle_binary!(
            "v1 i64x2",
            <X86V1 as Simd>::i64x2,
            i64,
            mulhi,
            |a, b| (((a as i128) * (b as i128)) >> 64) as i64,
            Tol::Exact
        );
        oracle_binary!(
            "v1 u64x2",
            <X86V1 as Simd>::u64x2,
            u64,
            mulhi,
            |a, b| (((a as u128) * (b as u128)) >> 64) as u64,
            Tol::Exact
        );
    }

    // Exercises the fixed bitwise-blendv saturating add/sub (sign-broadcast masks).
    #[test]
    fn saturating() {
        oracle_binary!(
            "v1 i32x4",
            <X86V1 as Simd>::i32x4,
            i32,
            saturating_add,
            |a, b| a.saturating_add(b),
            Tol::Exact
        );
        oracle_binary!(
            "v1 i32x4",
            <X86V1 as Simd>::i32x4,
            i32,
            saturating_sub,
            |a, b| a.saturating_sub(b),
            Tol::Exact
        );
        oracle_binary!(
            "v1 i64x2",
            <X86V1 as Simd>::i64x2,
            i64,
            saturating_add,
            |a, b| a.saturating_add(b),
            Tol::Exact
        );
        oracle_binary!(
            "v1 i64x2",
            <X86V1 as Simd>::i64x2,
            i64,
            saturating_sub,
            |a, b| a.saturating_sub(b),
            Tol::Exact
        );
        oracle_binary!(
            "v1 u32x4",
            <X86V1 as Simd>::u32x4,
            u32,
            saturating_add,
            |a, b| a.saturating_add(b),
            Tol::Exact
        );
        oracle_binary!(
            "v1 u32x4",
            <X86V1 as Simd>::u32x4,
            u32,
            saturating_sub,
            |a, b| a.saturating_sub(b),
            Tol::Exact
        );
        oracle_binary!(
            "v1 u64x2",
            <X86V1 as Simd>::u64x2,
            u64,
            saturating_add,
            |a, b| a.saturating_add(b),
            Tol::Exact
        );
        oracle_binary!(
            "v1 u64x2",
            <X86V1 as Simd>::u64x2,
            u64,
            saturating_sub,
            |a, b| a.saturating_sub(b),
            Tol::Exact
        );
    }

    #[test]
    fn bitscan() {
        for_lztz!("v1 i32x4", <X86V1 as Simd>::i32x4, i32);
        for_lztz!("v1 u32x4", <X86V1 as Simd>::u32x4, u32);
        for_lztz!("v1 i64x2", <X86V1 as Simd>::i64x2, i64);
        for_lztz!("v1 u64x2", <X86V1 as Simd>::u64x2, u64);
    }

    #[test]
    fn signed_extras() {
        for_signed!("v1 i32x4", <X86V1 as Simd>::i32x4, i32, i64);
        for_signed!("v1 i64x2", <X86V1 as Simd>::i64x2, i64, i128);
    }

    // Float polyfills with bit-exact Rust oracles. `fract` routes through the
    // magic-number `trunc` polyfill, so this covers the SSE2 rounding family.
    #[test]
    fn float_ops() {
        for_float!("v1 f32x4", <X86V1 as Simd>::f32x4, f32);
        for_float!("v1 f64x2", <X86V1 as Simd>::f64x2, f64);
    }

    // The SSE2 rounding polyfills, directly, against the Rust scalar ops.
    #[test]
    fn rounding() {
        oracle_unary!(
            "v1 f32x4",
            <X86V1 as Simd>::f32x4,
            f32,
            floor,
            |x: f32| x.floor(),
            Tol::Exact
        );
        oracle_unary!(
            "v1 f32x4",
            <X86V1 as Simd>::f32x4,
            f32,
            ceil,
            |x: f32| x.ceil(),
            Tol::Exact
        );
        oracle_unary!(
            "v1 f32x4",
            <X86V1 as Simd>::f32x4,
            f32,
            trunc,
            |x: f32| x.trunc(),
            Tol::Exact
        );
        oracle_unary!(
            "v1 f32x4",
            <X86V1 as Simd>::f32x4,
            f32,
            round,
            |x: f32| x.round_ties_even(),
            Tol::Exact
        );
        oracle_unary!(
            "v1 f64x2",
            <X86V1 as Simd>::f64x2,
            f64,
            floor,
            |x: f64| x.floor(),
            Tol::Exact
        );
        oracle_unary!(
            "v1 f64x2",
            <X86V1 as Simd>::f64x2,
            f64,
            ceil,
            |x: f64| x.ceil(),
            Tol::Exact
        );
        oracle_unary!(
            "v1 f64x2",
            <X86V1 as Simd>::f64x2,
            f64,
            trunc,
            |x: f64| x.trunc(),
            Tol::Exact
        );
        oracle_unary!(
            "v1 f64x2",
            <X86V1 as Simd>::f64x2,
            f64,
            round,
            |x: f64| x.round_ties_even(),
            Tol::Exact
        );
    }
}

// ===========================================================================
// Regression tests for the four polyfill defects the harness found - all now
// **fixed**, so these run in the default suite (no longer #[ignore]d).
//   P1 32-bit mulhi: `b` not shifted in _mm{,256}_mullhi_ep[iu]32x
//   P2 signed saturating add/sub: byte-granularity blendv mask in _mm_adds*_v2
//   P3 u64 lz/tz: 32-bit constant + count_ones copy-paste in U64x2 lz/tz
//   P4 u64 mullo: was todo!() - now reuses the sign-agnostic mul emulation
// See TESTING.md for root-cause analysis.
// ===========================================================================
mod fixed {
    use super::*;

    #[test]
    fn p1_mulhi32() {
        oracle_binary!(
            "v3 i32x4",
            <X86V3 as Simd>::i32x4,
            i32,
            mulhi,
            |a, b| (((a as i64) * (b as i64)) >> 32) as i32,
            Tol::Exact
        );
        oracle_binary!(
            "v3 u32x4",
            <X86V3 as Simd>::u32x4,
            u32,
            mulhi,
            |a, b| (((a as u64) * (b as u64)) >> 32) as u32,
            Tol::Exact
        );
        oracle_binary!(
            "v3 i32x8",
            <X86V3 as Simd>::i32x8,
            i32,
            mulhi,
            |a, b| (((a as i64) * (b as i64)) >> 32) as i32,
            Tol::Exact
        );
        oracle_binary!(
            "v3 u32x8",
            <X86V3 as Simd>::u32x8,
            u32,
            mulhi,
            |a, b| (((a as u64) * (b as u64)) >> 32) as u32,
            Tol::Exact
        );
        oracle_binary!(
            "v2 i32x4",
            <X86V2 as Simd>::i32x4,
            i32,
            mulhi,
            |a, b| (((a as i64) * (b as i64)) >> 32) as i32,
            Tol::Exact
        );
        oracle_binary!(
            "v2 u32x4",
            <X86V2 as Simd>::u32x4,
            u32,
            mulhi,
            |a, b| (((a as u64) * (b as u64)) >> 32) as u32,
            Tol::Exact
        );
    }

    #[test]
    fn p2_saturating_signed() {
        oracle_binary!(
            "v3 i32x4",
            <X86V3 as Simd>::i32x4,
            i32,
            saturating_add,
            |a, b| a.saturating_add(b),
            Tol::Exact
        );
        oracle_binary!(
            "v3 i32x4",
            <X86V3 as Simd>::i32x4,
            i32,
            saturating_sub,
            |a, b| a.saturating_sub(b),
            Tol::Exact
        );
        oracle_binary!(
            "v3 i64x2",
            <X86V3 as Simd>::i64x2,
            i64,
            saturating_add,
            |a, b| a.saturating_add(b),
            Tol::Exact
        );
        oracle_binary!(
            "v3 i64x2",
            <X86V3 as Simd>::i64x2,
            i64,
            saturating_sub,
            |a, b| a.saturating_sub(b),
            Tol::Exact
        );
        oracle_binary!(
            "v2 i32x4",
            <X86V2 as Simd>::i32x4,
            i32,
            saturating_add,
            |a, b| a.saturating_add(b),
            Tol::Exact
        );
        oracle_binary!(
            "v2 i64x2",
            <X86V2 as Simd>::i64x2,
            i64,
            saturating_sub,
            |a, b| a.saturating_sub(b),
            Tol::Exact
        );
    }

    #[test]
    fn p3_bitscan_u64() {
        oracle_unary!(
            "v3 u64x2",
            <X86V3 as Simd>::u64x2,
            u64,
            leading_zeros,
            |x| x.leading_zeros() as u64,
            Tol::Exact
        );
        oracle_unary!(
            "v3 u64x2",
            <X86V3 as Simd>::u64x2,
            u64,
            trailing_zeros,
            |x| x.trailing_zeros() as u64,
            Tol::Exact
        );
        oracle_unary!(
            "v2 u64x2",
            <X86V2 as Simd>::u64x2,
            u64,
            leading_zeros,
            |x| x.leading_zeros() as u64,
            Tol::Exact
        );
        oracle_unary!(
            "v2 u64x2",
            <X86V2 as Simd>::u64x2,
            u64,
            trailing_zeros,
            |x| x.trailing_zeros() as u64,
            Tol::Exact
        );
    }

    #[test]
    fn p4_mullo_u64() {
        oracle_binary!(
            "v3 u64x2",
            <X86V3 as Simd>::u64x2,
            u64,
            mullo,
            |a, b| a.wrapping_mul(b),
            Tol::Exact
        );
        oracle_binary!(
            "v3 u64x4",
            <X86V3 as Simd>::u64x4,
            u64,
            mullo,
            |a, b| a.wrapping_mul(b),
            Tol::Exact
        );
        oracle_binary!(
            "v2 u64x2",
            <X86V2 as Simd>::u64x2,
            u64,
            mullo,
            |a, b| a.wrapping_mul(b),
            Tol::Exact
        );
    }

    // P5: i32 `copysign` was implemented with `psignd`, which negates `lhs`
    // whenever `rhs` is negative *regardless of `lhs`'s own sign* - wrong for
    // every negative `lhs` (e.g. copysign(-3, -1) returned +3). True copysign
    // negates exactly where the signs differ. (i64 always used the xor-of-signs
    // form and was correct; pinned here too.)
    #[test]
    fn p5_copysign_int() {
        fn cs32(a: i32, b: i32) -> i32 {
            if (a < 0) != (b < 0) { a.wrapping_neg() } else { a }
        }
        fn cs64(a: i64, b: i64) -> i64 {
            if (a < 0) != (b < 0) { a.wrapping_neg() } else { a }
        }

        oracle_binary!("v1 i32x4", <X86V1 as Simd>::i32x4, i32, copysign, cs32, Tol::Exact);
        oracle_binary!("v2 i32x4", <X86V2 as Simd>::i32x4, i32, copysign, cs32, Tol::Exact);
        oracle_binary!("v3 i32x4", <X86V3 as Simd>::i32x4, i32, copysign, cs32, Tol::Exact);
        oracle_binary!("v3 i32x8", <X86V3 as Simd>::i32x8, i32, copysign, cs32, Tol::Exact);

        oracle_binary!("v1 i64x2", <X86V1 as Simd>::i64x2, i64, copysign, cs64, Tol::Exact);
        oracle_binary!("v2 i64x2", <X86V2 as Simd>::i64x2, i64, copysign, cs64, Tol::Exact);
        oracle_binary!("v3 i64x2", <X86V3 as Simd>::i64x2, i64, copysign, cs64, Tol::Exact);
        oracle_binary!("v3 i64x4", <X86V3 as Simd>::i64x4, i64, copysign, cs64, Tol::Exact);
    }

    // P6: the `u32x2 -> u64x2` zero-extending cast placed the u32 value in the
    // *high* dword of each u64 lane (`setr_epi32(0, v, ...)`), effectively
    // multiplying by 2^32. The value belongs in the low dword.
    #[test]
    fn p6_cast_u32x2_to_u64x2() {
        macro_rules! check {
            ($label:expr, $backend:ty) => {{
                use thermite::register::CastRegister;

                type Half = <$backend as Simd>::u32x2;
                type Full = <$backend as Simd>::u64x2;

                let mut rng = harness::rng();
                for input in harness::corpus::<u32>(2, &mut rng) {
                    let half = harness::make_array::<Half>(&input);
                    let full = <Full as CastRegister<Half>>::cast_from(half);

                    let got = harness::read::<Full>(&full);
                    let want: Vec<u64> = input.iter().map(|&x| x as u64).collect();

                    harness::assert_lanes_eq(
                        concat!($label, " [u32x2 -> u64x2 cast]"),
                        &[want.as_slice()],
                        &got,
                        &want,
                        Tol::Exact,
                    );
                }
            }};
        }

        check!("v1", X86V1);
        check!("v2", X86V2);
        check!("v3", X86V3);
    }
}
}

// wasm: exercise the same backend-generic polyfill macros on Wasm's native types
// (i32x4/u32x4/i64x2/u64x2 + f32x4/f64x2) - validates wasm's count_ones/leading_zeros/
// swap_bytes/reverse_bits/rotates/sra/avg/copysign/fract against scalar oracles.
#[cfg(target_arch = "wasm32")]
mod wasm {
use super::*;
use thermite::backend::wasm::Wasm;

mullo_for! {
    i32x4: Wasm, i32, "wasm i32x4"; i64x2: Wasm, i64, "wasm i64x2";
    u32x4: Wasm, u32, "wasm u32x4"; u64x2: Wasm, u64, "wasm u64x2";
}
popcount_for! {
    i32x4: Wasm, i32, "wasm i32x4"; u32x4: Wasm, u32, "wasm u32x4";
    i64x2: Wasm, i64, "wasm i64x2"; u64x2: Wasm, u64, "wasm u64x2";
}
bitperm_for! {
    i32x4: Wasm, i32, "wasm i32x4"; u32x4: Wasm, u32, "wasm u32x4";
    i64x2: Wasm, i64, "wasm i64x2"; u64x2: Wasm, u64, "wasm u64x2";
}

#[test]
fn lztz_signed_float() {
    for_lztz!("wasm i32x4", <Wasm as Simd>::i32x4, i32);
    for_lztz!("wasm u32x4", <Wasm as Simd>::u32x4, u32);
    for_lztz!("wasm i64x2", <Wasm as Simd>::i64x2, i64);
    for_lztz!("wasm u64x2", <Wasm as Simd>::u64x2, u64);
    for_signed!("wasm i32x4", <Wasm as Simd>::i32x4, i32, i64);
    for_signed!("wasm i64x2", <Wasm as Simd>::i64x2, i64, i128);
    for_float!("wasm f32x4", <Wasm as Simd>::f32x4, f32);
    for_float!("wasm f64x2", <Wasm as Simd>::f64x2, f64);
}
}
