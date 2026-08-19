//! Numeric `cast` (conversion) polyfill audit.
//!
//! Int<->float and width-changing conversions are heavily polyfilled
//! (`_mm*_cvt*`, `convert_*_limited`). The scalar backend's `cast_from` is
//! literally `value as _`, so each test is a differential against Rust's
//! built-in `as`, the documented "like `as`" contract for `cast`.
//!
//! `mod gate` is the always-green correctness gate. Float->int conversions are
//! kept in the **in-range, finite** domain there, matching `cast`'s documented
//! precondition (out-of-range/NaN lanes are backend-defined, x86 returning the
//! hardware "indefinite" integer). The total, `as`-exact op is
//! `saturating_cast`, verified over the raw corpus (NaN/inf/out-of-range
//! included) in each backend's `mod saturating`; those modules also pin that
//! `cast` f64->u64 is full-range truncating (it used to route to the rounding,
//! `[0, 2^52)`-only `_limited` polyfill that now only backs `fast_cast`).
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

// identity prep (int->int, int->float, float->float: scalar uses the same `as`,
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

/// One same-size `BitCastRegister` (byte reinterpret) pair, backend vs scalar.
macro_rules! bitpair {
    ($l:expr, $b:ty, $src:ident, $dst:ident, $se:ty) => {
        bitcast_diff!(
            $l,
            <$b as Simd>::$src,
            <$b as Simd>::$dst,
            <Scalar as Simd>::$src,
            <Scalar as Simd>::$dst,
            $se
        );
    };
}

// Float->int domain guards: keep strictly in-range & finite so truncation is
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

// Float->8/16-bit-int domain guards. The narrow path is `float -> i32 (trunc) -> low
// byte/word`, so it only matches scalar `as` (which saturates to the *target* int bounds)
// when the input already lies within the target type's range. Clamp into that range.
fn f32_to_i8_dom(x: f32) -> f32 {
    if x.is_finite() { x.clamp(-128.0, 127.0) } else { 0.0 }
}
fn f32_to_u8_dom(x: f32) -> f32 {
    if x.is_finite() { x.clamp(0.0, 255.0) } else { 0.0 }
}
fn f32_to_i16_dom(x: f32) -> f32 {
    if x.is_finite() { x.clamp(-32768.0, 32767.0) } else { 0.0 }
}
fn f32_to_u16_dom(x: f32) -> f32 {
    if x.is_finite() { x.clamp(0.0, 65535.0) } else { 0.0 }
}
fn f64_to_i8_dom(x: f64) -> f64 {
    if x.is_finite() { x.clamp(-128.0, 127.0) } else { 0.0 }
}
fn f64_to_u8_dom(x: f64) -> f64 {
    if x.is_finite() { x.clamp(0.0, 255.0) } else { 0.0 }
}
fn f64_to_i16_dom(x: f64) -> f64 {
    if x.is_finite() { x.clamp(-32768.0, 32767.0) } else { 0.0 }
}
fn f64_to_u16_dom(x: f64) -> f64 {
    if x.is_finite() { x.clamp(0.0, 65535.0) } else { 0.0 }
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

            // --- float->int, kept strictly in-range/finite ---
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
            // f64->u64 is not repeated here. It is covered exhaustively, along
            // with every other float->int pair at every lane count, by
            // `diff_cast_matrix.rs`. The rounding `_limited` epu64 polyfill backs
            // only `fast_cast`, so the in-range differential passes.

            // --- rung 3: 8/16-bit int <-> f32/f64 direct casts ---
            // widen int -> float is value-preserving and exact (every i8/u8/i16/u16 is
            // exactly representable in f32 and f64); identity prep, Tol::Exact.
            #[test]
            fn int8_to_f32_widen() {
                cpair!($tag, $b, i8x2, f32x2, i8, id!(i8), Tol::Exact);
                cpair!($tag, $b, u8x2, f32x2, u8, id!(u8), Tol::Exact);
                cpair!($tag, $b, i8x4, f32x4, i8, id!(i8), Tol::Exact);
                cpair!($tag, $b, u8x4, f32x4, u8, id!(u8), Tol::Exact);
                cpair!($tag, $b, i8x8, f32x8, i8, id!(i8), Tol::Exact);
                cpair!($tag, $b, u8x8, f32x8, u8, id!(u8), Tol::Exact);
                cpair!($tag, $b, i8x16, f32x16, i8, id!(i8), Tol::Exact);
                cpair!($tag, $b, u8x16, f32x16, u8, id!(u8), Tol::Exact);
            }
            #[test]
            fn int8_to_f64_widen() {
                cpair!($tag, $b, i8x2, f64x2, i8, id!(i8), Tol::Exact);
                cpair!($tag, $b, u8x2, f64x2, u8, id!(u8), Tol::Exact);
                cpair!($tag, $b, i8x4, f64x4, i8, id!(i8), Tol::Exact);
                cpair!($tag, $b, u8x4, f64x4, u8, id!(u8), Tol::Exact);
                cpair!($tag, $b, i8x8, f64x8, i8, id!(i8), Tol::Exact);
                cpair!($tag, $b, u8x8, f64x8, u8, id!(u8), Tol::Exact);
                cpair!($tag, $b, i8x16, f64x16, i8, id!(i8), Tol::Exact);
                cpair!($tag, $b, u8x16, f64x16, u8, id!(u8), Tol::Exact);
            }
            #[test]
            fn int16_to_f32_widen() {
                cpair!($tag, $b, i16x2, f32x2, i16, id!(i16), Tol::Exact);
                cpair!($tag, $b, u16x2, f32x2, u16, id!(u16), Tol::Exact);
                cpair!($tag, $b, i16x4, f32x4, i16, id!(i16), Tol::Exact);
                cpair!($tag, $b, u16x4, f32x4, u16, id!(u16), Tol::Exact);
                cpair!($tag, $b, i16x8, f32x8, i16, id!(i16), Tol::Exact);
                cpair!($tag, $b, u16x8, f32x8, u16, id!(u16), Tol::Exact);
                cpair!($tag, $b, i16x16, f32x16, i16, id!(i16), Tol::Exact);
                cpair!($tag, $b, u16x16, f32x16, u16, id!(u16), Tol::Exact);
            }
            #[test]
            fn int16_to_f64_widen() {
                cpair!($tag, $b, i16x2, f64x2, i16, id!(i16), Tol::Exact);
                cpair!($tag, $b, u16x2, f64x2, u16, id!(u16), Tol::Exact);
                cpair!($tag, $b, i16x4, f64x4, i16, id!(i16), Tol::Exact);
                cpair!($tag, $b, u16x4, f64x4, u16, id!(u16), Tol::Exact);
                cpair!($tag, $b, i16x8, f64x8, i16, id!(i16), Tol::Exact);
                cpair!($tag, $b, u16x8, f64x8, u16, id!(u16), Tol::Exact);
                cpair!($tag, $b, i16x16, f64x16, i16, id!(i16), Tol::Exact);
                cpair!($tag, $b, u16x16, f64x16, u16, id!(u16), Tol::Exact);
            }

            // narrow float -> 8/16-bit int, kept strictly in the target type's range
            // (the contract is 'like as' only in-range, and out-of-range/NaN diverges).
            #[test]
            fn f32_to_int8_inrange() {
                cpair!(
                    $tag,
                    $b,
                    f32x2,
                    i8x2,
                    f32,
                    f32_to_i8_dom as fn(f32) -> f32,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f32x2,
                    u8x2,
                    f32,
                    f32_to_u8_dom as fn(f32) -> f32,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f32x4,
                    i8x4,
                    f32,
                    f32_to_i8_dom as fn(f32) -> f32,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f32x4,
                    u8x4,
                    f32,
                    f32_to_u8_dom as fn(f32) -> f32,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f32x8,
                    i8x8,
                    f32,
                    f32_to_i8_dom as fn(f32) -> f32,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f32x8,
                    u8x8,
                    f32,
                    f32_to_u8_dom as fn(f32) -> f32,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f32x16,
                    i8x16,
                    f32,
                    f32_to_i8_dom as fn(f32) -> f32,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f32x16,
                    u8x16,
                    f32,
                    f32_to_u8_dom as fn(f32) -> f32,
                    Tol::Exact
                );
            }
            #[test]
            fn f64_to_int8_inrange() {
                cpair!(
                    $tag,
                    $b,
                    f64x2,
                    i8x2,
                    f64,
                    f64_to_i8_dom as fn(f64) -> f64,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f64x2,
                    u8x2,
                    f64,
                    f64_to_u8_dom as fn(f64) -> f64,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f64x4,
                    i8x4,
                    f64,
                    f64_to_i8_dom as fn(f64) -> f64,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f64x4,
                    u8x4,
                    f64,
                    f64_to_u8_dom as fn(f64) -> f64,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f64x8,
                    i8x8,
                    f64,
                    f64_to_i8_dom as fn(f64) -> f64,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f64x8,
                    u8x8,
                    f64,
                    f64_to_u8_dom as fn(f64) -> f64,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f64x16,
                    i8x16,
                    f64,
                    f64_to_i8_dom as fn(f64) -> f64,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f64x16,
                    u8x16,
                    f64,
                    f64_to_u8_dom as fn(f64) -> f64,
                    Tol::Exact
                );
            }
            #[test]
            fn f32_to_int16_inrange() {
                cpair!(
                    $tag,
                    $b,
                    f32x2,
                    i16x2,
                    f32,
                    f32_to_i16_dom as fn(f32) -> f32,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f32x2,
                    u16x2,
                    f32,
                    f32_to_u16_dom as fn(f32) -> f32,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f32x4,
                    i16x4,
                    f32,
                    f32_to_i16_dom as fn(f32) -> f32,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f32x4,
                    u16x4,
                    f32,
                    f32_to_u16_dom as fn(f32) -> f32,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f32x8,
                    i16x8,
                    f32,
                    f32_to_i16_dom as fn(f32) -> f32,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f32x8,
                    u16x8,
                    f32,
                    f32_to_u16_dom as fn(f32) -> f32,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f32x16,
                    i16x16,
                    f32,
                    f32_to_i16_dom as fn(f32) -> f32,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f32x16,
                    u16x16,
                    f32,
                    f32_to_u16_dom as fn(f32) -> f32,
                    Tol::Exact
                );
            }
            #[test]
            fn f64_to_int16_inrange() {
                cpair!(
                    $tag,
                    $b,
                    f64x2,
                    i16x2,
                    f64,
                    f64_to_i16_dom as fn(f64) -> f64,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f64x2,
                    u16x2,
                    f64,
                    f64_to_u16_dom as fn(f64) -> f64,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f64x4,
                    i16x4,
                    f64,
                    f64_to_i16_dom as fn(f64) -> f64,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f64x4,
                    u16x4,
                    f64,
                    f64_to_u16_dom as fn(f64) -> f64,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f64x8,
                    i16x8,
                    f64,
                    f64_to_i16_dom as fn(f64) -> f64,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f64x8,
                    u16x8,
                    f64,
                    f64_to_u16_dom as fn(f64) -> f64,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f64x16,
                    i16x16,
                    f64,
                    f64_to_i16_dom as fn(f64) -> f64,
                    Tol::Exact
                );
                cpair!(
                    $tag,
                    $b,
                    f64x16,
                    u16x16,
                    f64,
                    f64_to_u16_dom as fn(f64) -> f64,
                    Tol::Exact
                );
            }

            // --- same-size i <-> u bitcasts (byte reinterpret) for the 8/16-bit slots ---
            #[test]
            fn bitcast_int8() {
                bitpair!($tag, $b, i8x2, u8x2, i8);
                bitpair!($tag, $b, u8x2, i8x2, u8);
                bitpair!($tag, $b, i8x4, u8x4, i8);
                bitpair!($tag, $b, u8x4, i8x4, u8);
                bitpair!($tag, $b, i8x8, u8x8, i8);
                bitpair!($tag, $b, u8x8, i8x8, u8);
                bitpair!($tag, $b, i8x16, u8x16, i8);
                bitpair!($tag, $b, u8x16, i8x16, u8);
            }
            #[test]
            fn bitcast_int16() {
                bitpair!($tag, $b, i16x2, u16x2, i16);
                bitpair!($tag, $b, u16x2, i16x2, u16);
                bitpair!($tag, $b, i16x4, u16x4, i16);
                bitpair!($tag, $b, u16x4, i16x4, u16);
                bitpair!($tag, $b, i16x8, u16x8, i16);
                bitpair!($tag, $b, u16x8, i16x8, u16);
                bitpair!($tag, $b, i16x16, u16x16, i16);
                bitpair!($tag, $b, u16x16, i16x16, u16);
            }
        }
    };
}

// Generic reachability: confirms the `Simd` trait itself carries the cross-slot
// `BitCastRegister<sibling>` bound for the 8/16-bit families (so generic `S: Simd` code can
// name the i <-> u bitcast without pinning a concrete backend or going through `Register::Unsigned`).
#[allow(dead_code)]
fn assert_generic_bitcast_bounds<S: Simd>() {
    fn needs<A: thermite::register::CoreRegister, B: thermite::register::BitCastRegister<A>>() {}
    needs::<S::i8x4, S::u8x4>();
    needs::<S::u8x4, S::i8x4>();
    needs::<S::i8x16, S::u8x16>();
    needs::<S::i16x8, S::u16x8>();
    needs::<S::u16x16, S::i16x16>();
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

    // Out-of-range / NaN float->int: `cast` documents backend-defined results
    // there (x86 returns the hardware "indefinite" integer). The total,
    // `as`-exact op is `saturating_cast`, verified here over the RAW corpus
    // (NaN, infinities, out-of-range included) against the scalar oracle.
    mod saturating {
        use super::*;

        macro_rules! sat_suite {
            ($b:ty, $tag:expr) => {
                sat_cast_diff!(
                    concat!($tag, " f32x4->i32x4 sat"),
                    <$b as Simd>::f32x4,
                    <$b as Simd>::i32x4,
                    <Scalar as Simd>::f32x4,
                    <Scalar as Simd>::i32x4,
                    f32
                );
                sat_cast_diff!(
                    concat!($tag, " f32x8->i32x8 sat"),
                    <$b as Simd>::f32x8,
                    <$b as Simd>::i32x8,
                    <Scalar as Simd>::f32x8,
                    <Scalar as Simd>::i32x8,
                    f32
                );
                sat_cast_diff!(
                    concat!($tag, " f32x4->u32x4 sat"),
                    <$b as Simd>::f32x4,
                    <$b as Simd>::u32x4,
                    <Scalar as Simd>::f32x4,
                    <Scalar as Simd>::u32x4,
                    f32
                );
                sat_cast_diff!(
                    concat!($tag, " f64x2->i64x2 sat"),
                    <$b as Simd>::f64x2,
                    <$b as Simd>::i64x2,
                    <Scalar as Simd>::f64x2,
                    <Scalar as Simd>::i64x2,
                    f64
                );
                sat_cast_diff!(
                    concat!($tag, " f64x4->i64x4 sat"),
                    <$b as Simd>::f64x4,
                    <$b as Simd>::i64x4,
                    <Scalar as Simd>::f64x4,
                    <Scalar as Simd>::i64x4,
                    f64
                );
                sat_cast_diff!(
                    concat!($tag, " f64x4->u64x4 sat"),
                    <$b as Simd>::f64x4,
                    <$b as Simd>::u64x4,
                    <Scalar as Simd>::f64x4,
                    <Scalar as Simd>::u64x4,
                    f64
                );

                // cross-width compositions: narrow via same-width int...
                sat_cast_diff!(
                    concat!($tag, " f32x8->i16x8 sat"),
                    <$b as Simd>::f32x8,
                    <$b as Simd>::i16x8,
                    <Scalar as Simd>::f32x8,
                    <Scalar as Simd>::i16x8,
                    f32
                );
                sat_cast_diff!(
                    concat!($tag, " f32x4->i8x4 sat"),
                    <$b as Simd>::f32x4,
                    <$b as Simd>::i8x4,
                    <Scalar as Simd>::f32x4,
                    <Scalar as Simd>::i8x4,
                    f32
                );
                sat_cast_diff!(
                    concat!($tag, " f64x4->i32x4 sat"),
                    <$b as Simd>::f64x4,
                    <$b as Simd>::i32x4,
                    <Scalar as Simd>::f64x4,
                    <Scalar as Simd>::i32x4,
                    f64
                );
                sat_cast_diff!(
                    concat!($tag, " f64x4->u8x4 sat"),
                    <$b as Simd>::f64x4,
                    <$b as Simd>::u8x4,
                    <Scalar as Simd>::f64x4,
                    <Scalar as Simd>::u8x4,
                    f64
                );
                sat_cast_diff!(
                    concat!($tag, " f64x16->i16x16 sat"),
                    <$b as Simd>::f64x16,
                    <$b as Simd>::i16x16,
                    <Scalar as Simd>::f64x16,
                    <Scalar as Simd>::i16x16,
                    f64
                );
                // ...and the f32 -> 64-bit widen-then-saturate arm
                sat_cast_diff!(
                    concat!($tag, " f32x4->u64x4 sat"),
                    <$b as Simd>::f32x4,
                    <$b as Simd>::u64x4,
                    <Scalar as Simd>::f32x4,
                    <Scalar as Simd>::u64x4,
                    f32
                );
                sat_cast_diff!(
                    concat!($tag, " f32x8->i64x8 sat"),
                    <$b as Simd>::f32x8,
                    <$b as Simd>::i64x8,
                    <Scalar as Simd>::f32x8,
                    <Scalar as Simd>::i64x8,
                    f32
                );
            };
        }

        // Bound smoke: `S: Simd` alone must be enough for generic saturating
        // float -> int casts (the pairs are bound on the `Simd` trait's slots),
        // including the reduced x2 width.
        fn generic_over_simd<S: thermite::simd::Simd>() {
            use thermite::prelude::*;

            let v = Vector::<S::f32x8>::splat(f32::NAN).saturating_cast::<Vector<S::i32x8>>();
            assert_eq!(v.extract::<0>(), 0);

            let v = Vector::<S::f64x4>::splat(1e300).saturating_cast::<Vector<S::u64x4>>();
            assert_eq!(v.extract::<0>(), u64::MAX);

            let v = Vector::<S::f32x2>::splat(-1e10).saturating_cast::<Vector<S::i32x2>>();
            assert_eq!(v.extract::<0>(), i32::MIN);

            // cross-width pairs resolve from the Simd bounds too
            let v = Vector::<S::f64x4>::splat(-1e300).saturating_cast::<Vector<S::i8x4>>();
            assert_eq!(v.extract::<0>(), i8::MIN);

            let v = Vector::<S::f32x4>::splat(1e30).saturating_cast::<Vector<S::u64x4>>();
            assert_eq!(v.extract::<0>(), u64::MAX);
        }

        #[test]
        fn saturating_cast_generic_bounds() {
            generic_over_simd::<Scalar>();
            generic_over_simd::<X86V1>();
            generic_over_simd::<X86V2>();
            generic_over_simd::<X86V3>();
        }

        #[test]
        fn v1_float_to_int_saturates() {
            sat_suite!(X86V1, "x86_v1");
        }

        #[test]
        fn v2_float_to_int_saturates() {
            sat_suite!(X86V2, "x86_v2");
        }

        #[test]
        fn v3_float_to_int_saturates() {
            sat_suite!(X86V3, "x86_v3");
        }

        // `cast` f64->u64 is full-range truncating now (previously it routed to
        // the `_limited` magic-add polyfill, which ROUNDS and only covers
        // [0, 2^52) - the fractional and the beyond-2^52 case both regressed to
        // `as` behavior). `fast_cast` keeps the 2-op limited trick.
        #[test]
        fn f64_to_u64_cast_truncates() {
            // fractional in-range values must truncate like `as`, not round
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
            // the full in-range domain, including beyond the old 2^52 limit
            cast_diff!(
                "x86_v3 f64x4->u64x4 full range",
                <X86V3 as Simd>::f64x4,
                <X86V3 as Simd>::u64x4,
                <Scalar as Simd>::f64x4,
                <Scalar as Simd>::u64x4,
                f64,
                |x: f64| if x.is_finite() { x.abs().clamp(0.0, 1.8e19) } else { 2.7 },
                Tol::Exact
            );
            cast_diff!(
                "x86_v1 f64x2->u64x2 full range",
                <X86V1 as Simd>::f64x2,
                <X86V1 as Simd>::u64x2,
                <Scalar as Simd>::f64x2,
                <Scalar as Simd>::u64x2,
                f64,
                |x: f64| if x.is_finite() { x.abs().clamp(0.0, 1.8e19) } else { 2.7 },
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

    // wasm's `cast` is already total and `as`-exact in both directions
    // (`*_trunc_sat_*` for f32, per-lane scalar `as` for f64), so the raw
    // corpus (NaN, infinities, out-of-range) must match the scalar oracle.
    // Only `fast_cast` (relaxed trunc / `_limited` magic) is narrow-domain.
    mod saturating {
        use super::*;

        #[test]
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
        fn f64_to_u64_cast_truncates() {
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

        #[test]
        fn float_to_int_saturates() {
            sat_cast_diff!(
                "wasm f32x4->i32x4 sat",
                <Wasm as Simd>::f32x4,
                <Wasm as Simd>::i32x4,
                <Scalar as Simd>::f32x4,
                <Scalar as Simd>::i32x4,
                f32
            );
            sat_cast_diff!(
                "wasm f32x4->u32x4 sat",
                <Wasm as Simd>::f32x4,
                <Wasm as Simd>::u32x4,
                <Scalar as Simd>::f32x4,
                <Scalar as Simd>::u32x4,
                f32
            );
            sat_cast_diff!(
                "wasm f64x2->i64x2 sat",
                <Wasm as Simd>::f64x2,
                <Wasm as Simd>::i64x2,
                <Scalar as Simd>::f64x2,
                <Scalar as Simd>::i64x2,
                f64
            );
            sat_cast_diff!(
                "wasm f64x2->u64x2 sat",
                <Wasm as Simd>::f64x2,
                <Wasm as Simd>::u64x2,
                <Scalar as Simd>::f64x2,
                <Scalar as Simd>::u64x2,
                f64
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use thermite::backend::neon::Neon;

    mod gate {
        use super::*;
        cast_suite!(neon, Neon, "neon");
    }

    // Out-of-range / NaN float->int: scalar saturates (Rust `as`), the wasm hardware
    // path returns the "indefinite" integer (i64::MIN / i32::MIN). Documented
    // divergence, not auto-failed.
    // (inherited from the wasm section, revisit for NEON)
    mod divergence {
        use super::*;

        #[test]
        #[ignore = "DIVERGENCE: out-of-range/NaN float->int returns the hardware \
                indefinite integer instead of saturating like `as` (scalar). \
                The general `cast` contract is 'like as'; backend needs a clamp or \
                the `_limited` precondition must be documented."]
        fn float_to_int_out_of_range() {
            cast_diff!(
                "neon f64x4->i64x4 OOR",
                <Neon as Simd>::f64x4,
                <Neon as Simd>::i64x4,
                <Scalar as Simd>::f64x4,
                <Scalar as Simd>::i64x4,
                f64,
                |x| x,
                Tol::Exact
            );
            cast_diff!(
                "neon f32x4->i32x4 OOR",
                <Neon as Simd>::f32x4,
                <Neon as Simd>::i32x4,
                <Scalar as Simd>::f32x4,
                <Scalar as Simd>::i32x4,
                f32,
                |x| x,
                Tol::Exact
            );
        }

        #[test]
        #[ignore = "DIVERGENCE: f64->u64 `cast` routes to a `_limited` polyfill \
                which (a) only works on [0, 2^52) and (b) ROUNDS (adds 2^52) \
                instead of truncating like `as` - so e.g. 2.7_f64 as u64 == 2 \
                but the cast yields 3, and values ≥ 2^52 are corrupted. \
                f64->i64 is full-range-correct and truncating; f64->u64 needs an \
                equivalent path or a documented precondition."]
        fn f64_to_u64_nonconforming() {
            // (b) rounds vs truncates, even for tiny in-range values.
            cast_diff!(
                "neon f64x4->u64x4 frac",
                <Neon as Simd>::f64x4,
                <Neon as Simd>::u64x4,
                <Scalar as Simd>::f64x4,
                <Scalar as Simd>::u64x4,
                f64,
                |x: f64| if x.is_finite() { (x.abs() % 1000.0) + 0.7 } else { 2.7 },
                Tol::Exact
            );
        }
    }
}
