//! Numeric `cast` (conversion) polyfill audit.
//!
//! Int<->float and width-changing conversions are heavily polyfilled
//! (`_mm*_cvt*`, `convert_*_limited`). The scalar backend's `cast_from` is
//! literally `value as _`, so each test is a differential against Rust's
//! built-in `as`, the documented "like `as`" contract for `cast`.
//!
//! The always-green correctness gate keeps float->int conversions in the
//! **in-range, finite** domain, matching `cast`'s documented precondition
//! (out-of-range/NaN lanes are backend-defined, x86 returning the hardware
//! "indefinite" integer). The total, `as`-exact op is `saturating_cast`,
//! verified over the raw corpus (NaN/inf/out-of-range included) in the
//! `sat_*` tests. Those also pin that `cast` f64->u64 is full-range truncating
//! (it used to route to the rounding, `[0, 2^52)`-only `_limited` polyfill that
//! now only backs `fast_cast`).
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

/// One cast pair, backend `$S` vs scalar, for a given (src, dst).
macro_rules! cpair {
    ($S:ty, $src:ident, $dst:ident, $se:ty, $prep:expr, $tol:expr) => {
        cast_diff!(
            harness::label::<$S>(concat!(stringify!($src), "->", stringify!($dst))),
            <$S as Simd>::$src,
            <$S as Simd>::$dst,
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
    ($S:ty, $src:ident, $dst:ident, $se:ty) => {
        bitcast_diff!(
            harness::label::<$S>(concat!(stringify!($src), "->", stringify!($dst))),
            <$S as Simd>::$src,
            <$S as Simd>::$dst,
            <Scalar as Simd>::$src,
            <Scalar as Simd>::$dst,
            $se
        );
    };
}

/// One float -> int `saturating_cast` pair over the RAW corpus.
macro_rules! spair {
    ($S:ty, $src:ident, $dst:ident, $se:ty) => {
        sat_cast_diff!(
            harness::label::<$S>(concat!(stringify!($src), "->", stringify!($dst), " sat")),
            <$S as Simd>::$src,
            <$S as Simd>::$dst,
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

for_each_backend! {
    // --- element swaps, same lane width (reinterpret as number) ---
    fn swaps_32<S: Simd>() {
        cpair!(S, i32x4, u32x4, i32, id!(i32), Tol::Exact);
        cpair!(S, u32x4, i32x4, u32, id!(u32), Tol::Exact);
        cpair!(S, i32x4, f32x4, i32, id!(i32), Tol::Exact);
        cpair!(S, u32x4, f32x4, u32, id!(u32), Tol::Exact);
        cpair!(S, i32x8, f32x8, i32, id!(i32), Tol::Exact);
        cpair!(S, u32x8, f32x8, u32, id!(u32), Tol::Exact);
        cpair!(S, i32x16, f32x16, i32, id!(i32), Tol::Exact);
        cpair!(S, u32x16, f32x16, u32, id!(u32), Tol::Exact);
    }
    fn swaps_64<S: Simd>() {
        cpair!(S, i64x2, u64x2, i64, id!(i64), Tol::Exact);
        cpair!(S, u64x2, i64x2, u64, id!(u64), Tol::Exact);
        cpair!(S, i64x2, f64x2, i64, id!(i64), Tol::Exact);
        cpair!(S, u64x2, f64x2, u64, id!(u64), Tol::Exact);
        cpair!(S, i64x4, f64x4, i64, id!(i64), Tol::Exact);
        cpair!(S, u64x4, f64x4, u64, id!(u64), Tol::Exact);
        cpair!(S, i64x8, f64x8, i64, id!(i64), Tol::Exact);
        cpair!(S, u64x8, f64x8, u64, id!(u64), Tol::Exact);
    }

    // --- width-changing int/float conversions ---
    fn widths<S: Simd>() {
        cpair!(S, i32x4, i64x4, i32, id!(i32), Tol::Exact); // widen
        cpair!(S, i64x4, i32x4, i64, id!(i64), Tol::Exact); // narrow
        cpair!(S, u32x4, u64x4, u32, id!(u32), Tol::Exact);
        cpair!(S, f32x4, f64x4, f32, id!(f32), Tol::Exact);
        cpair!(S, f64x4, f32x4, f64, id!(f64), Tol::Exact);
        cpair!(S, i32x8, i64x8, i32, id!(i32), Tol::Exact);
        cpair!(S, i64x8, i32x8, i64, id!(i64), Tol::Exact);
        cpair!(S, u32x8, u64x8, u32, id!(u32), Tol::Exact);
        cpair!(S, f32x8, f64x8, f32, id!(f32), Tol::Exact);
        cpair!(S, f64x8, f32x8, f64, id!(f64), Tol::Exact);
    }

    // --- float->int, kept strictly in-range/finite ---
    // f64->u64 is not repeated here. It is covered exhaustively, along
    // with every other float->int pair at every lane count, by
    // `diff_cast_matrix.rs`.
    fn f32_to_i32_inrange<S: Simd>() {
        cpair!(S, f32x4, i32x4, f32, to_i32_dom, Tol::Exact);
        cpair!(S, f32x8, i32x8, f32, to_i32_dom, Tol::Exact);
        cpair!(S, f32x16, i32x16, f32, to_i32_dom, Tol::Exact);
    }
    fn f32_to_u32_inrange<S: Simd>() {
        cpair!(S, f32x4, u32x4, f32, to_u32_dom, Tol::Exact);
        cpair!(S, f32x8, u32x8, f32, to_u32_dom, Tol::Exact);
        cpair!(S, f32x16, u32x16, f32, to_u32_dom, Tol::Exact);
    }
    fn f64_to_i64_inrange<S: Simd>() {
        cpair!(S, f64x2, i64x2, f64, to_i64_dom, Tol::Exact);
        cpair!(S, f64x4, i64x4, f64, to_i64_dom, Tol::Exact);
        cpair!(S, f64x8, i64x8, f64, to_i64_dom, Tol::Exact);
    }

    // --- rung 3: 8/16-bit int <-> f32/f64 direct casts ---
    // widen int -> float is value-preserving and exact (every i8/u8/i16/u16 is
    // exactly representable in f32 and f64). Identity prep, Tol::Exact.
    fn int8_to_f32_widen<S: Simd>() {
        cpair!(S, i8x2, f32x2, i8, id!(i8), Tol::Exact);
        cpair!(S, u8x2, f32x2, u8, id!(u8), Tol::Exact);
        cpair!(S, i8x4, f32x4, i8, id!(i8), Tol::Exact);
        cpair!(S, u8x4, f32x4, u8, id!(u8), Tol::Exact);
        cpair!(S, i8x8, f32x8, i8, id!(i8), Tol::Exact);
        cpair!(S, u8x8, f32x8, u8, id!(u8), Tol::Exact);
        cpair!(S, i8x16, f32x16, i8, id!(i8), Tol::Exact);
        cpair!(S, u8x16, f32x16, u8, id!(u8), Tol::Exact);
    }
    fn int8_to_f64_widen<S: Simd>() {
        cpair!(S, i8x2, f64x2, i8, id!(i8), Tol::Exact);
        cpair!(S, u8x2, f64x2, u8, id!(u8), Tol::Exact);
        cpair!(S, i8x4, f64x4, i8, id!(i8), Tol::Exact);
        cpair!(S, u8x4, f64x4, u8, id!(u8), Tol::Exact);
        cpair!(S, i8x8, f64x8, i8, id!(i8), Tol::Exact);
        cpair!(S, u8x8, f64x8, u8, id!(u8), Tol::Exact);
        cpair!(S, i8x16, f64x16, i8, id!(i8), Tol::Exact);
        cpair!(S, u8x16, f64x16, u8, id!(u8), Tol::Exact);
    }
    fn int16_to_f32_widen<S: Simd>() {
        cpair!(S, i16x2, f32x2, i16, id!(i16), Tol::Exact);
        cpair!(S, u16x2, f32x2, u16, id!(u16), Tol::Exact);
        cpair!(S, i16x4, f32x4, i16, id!(i16), Tol::Exact);
        cpair!(S, u16x4, f32x4, u16, id!(u16), Tol::Exact);
        cpair!(S, i16x8, f32x8, i16, id!(i16), Tol::Exact);
        cpair!(S, u16x8, f32x8, u16, id!(u16), Tol::Exact);
        cpair!(S, i16x16, f32x16, i16, id!(i16), Tol::Exact);
        cpair!(S, u16x16, f32x16, u16, id!(u16), Tol::Exact);
    }
    fn int16_to_f64_widen<S: Simd>() {
        cpair!(S, i16x2, f64x2, i16, id!(i16), Tol::Exact);
        cpair!(S, u16x2, f64x2, u16, id!(u16), Tol::Exact);
        cpair!(S, i16x4, f64x4, i16, id!(i16), Tol::Exact);
        cpair!(S, u16x4, f64x4, u16, id!(u16), Tol::Exact);
        cpair!(S, i16x8, f64x8, i16, id!(i16), Tol::Exact);
        cpair!(S, u16x8, f64x8, u16, id!(u16), Tol::Exact);
        cpair!(S, i16x16, f64x16, i16, id!(i16), Tol::Exact);
        cpair!(S, u16x16, f64x16, u16, id!(u16), Tol::Exact);
    }

    // narrow float -> 8/16-bit int, kept strictly in the target type's range
    // (the contract is 'like as' only in-range, and out-of-range/NaN diverges).
    fn f32_to_int8_inrange<S: Simd>() {
        cpair!(S, f32x2, i8x2, f32, f32_to_i8_dom, Tol::Exact);
        cpair!(S, f32x2, u8x2, f32, f32_to_u8_dom, Tol::Exact);
        cpair!(S, f32x4, i8x4, f32, f32_to_i8_dom, Tol::Exact);
        cpair!(S, f32x4, u8x4, f32, f32_to_u8_dom, Tol::Exact);
        cpair!(S, f32x8, i8x8, f32, f32_to_i8_dom, Tol::Exact);
        cpair!(S, f32x8, u8x8, f32, f32_to_u8_dom, Tol::Exact);
        cpair!(S, f32x16, i8x16, f32, f32_to_i8_dom, Tol::Exact);
        cpair!(S, f32x16, u8x16, f32, f32_to_u8_dom, Tol::Exact);
    }
    fn f64_to_int8_inrange<S: Simd>() {
        cpair!(S, f64x2, i8x2, f64, f64_to_i8_dom, Tol::Exact);
        cpair!(S, f64x2, u8x2, f64, f64_to_u8_dom, Tol::Exact);
        cpair!(S, f64x4, i8x4, f64, f64_to_i8_dom, Tol::Exact);
        cpair!(S, f64x4, u8x4, f64, f64_to_u8_dom, Tol::Exact);
        cpair!(S, f64x8, i8x8, f64, f64_to_i8_dom, Tol::Exact);
        cpair!(S, f64x8, u8x8, f64, f64_to_u8_dom, Tol::Exact);
        cpair!(S, f64x16, i8x16, f64, f64_to_i8_dom, Tol::Exact);
        cpair!(S, f64x16, u8x16, f64, f64_to_u8_dom, Tol::Exact);
    }
    fn f32_to_int16_inrange<S: Simd>() {
        cpair!(S, f32x2, i16x2, f32, f32_to_i16_dom, Tol::Exact);
        cpair!(S, f32x2, u16x2, f32, f32_to_u16_dom, Tol::Exact);
        cpair!(S, f32x4, i16x4, f32, f32_to_i16_dom, Tol::Exact);
        cpair!(S, f32x4, u16x4, f32, f32_to_u16_dom, Tol::Exact);
        cpair!(S, f32x8, i16x8, f32, f32_to_i16_dom, Tol::Exact);
        cpair!(S, f32x8, u16x8, f32, f32_to_u16_dom, Tol::Exact);
        cpair!(S, f32x16, i16x16, f32, f32_to_i16_dom, Tol::Exact);
        cpair!(S, f32x16, u16x16, f32, f32_to_u16_dom, Tol::Exact);
    }
    fn f64_to_int16_inrange<S: Simd>() {
        cpair!(S, f64x2, i16x2, f64, f64_to_i16_dom, Tol::Exact);
        cpair!(S, f64x2, u16x2, f64, f64_to_u16_dom, Tol::Exact);
        cpair!(S, f64x4, i16x4, f64, f64_to_i16_dom, Tol::Exact);
        cpair!(S, f64x4, u16x4, f64, f64_to_u16_dom, Tol::Exact);
        cpair!(S, f64x8, i16x8, f64, f64_to_i16_dom, Tol::Exact);
        cpair!(S, f64x8, u16x8, f64, f64_to_u16_dom, Tol::Exact);
        cpair!(S, f64x16, i16x16, f64, f64_to_i16_dom, Tol::Exact);
        cpair!(S, f64x16, u16x16, f64, f64_to_u16_dom, Tol::Exact);
    }

    // --- same-size i <-> u bitcasts (byte reinterpret) for the 8/16-bit slots ---
    fn bitcast_int8<S: Simd>() {
        bitpair!(S, i8x2, u8x2, i8);
        bitpair!(S, u8x2, i8x2, u8);
        bitpair!(S, i8x4, u8x4, i8);
        bitpair!(S, u8x4, i8x4, u8);
        bitpair!(S, i8x8, u8x8, i8);
        bitpair!(S, u8x8, i8x8, u8);
        bitpair!(S, i8x16, u8x16, i8);
        bitpair!(S, u8x16, i8x16, u8);
    }
    fn bitcast_int16<S: Simd>() {
        bitpair!(S, i16x2, u16x2, i16);
        bitpair!(S, u16x2, i16x2, u16);
        bitpair!(S, i16x4, u16x4, i16);
        bitpair!(S, u16x4, i16x4, u16);
        bitpair!(S, i16x8, u16x8, i16);
        bitpair!(S, u16x8, i16x8, u16);
        bitpair!(S, i16x16, u16x16, i16);
        bitpair!(S, u16x16, i16x16, u16);
    }

    // Out-of-range / NaN float->int: `cast` documents backend-defined results
    // there (x86 returns the hardware "indefinite" integer). The total,
    // `as`-exact op is `saturating_cast`, verified here over the RAW corpus
    // (NaN, infinities, out-of-range included) against the scalar oracle.
    fn sat_float_to_int<S: Simd>() {
        spair!(S, f32x4, i32x4, f32);
        spair!(S, f32x8, i32x8, f32);
        spair!(S, f32x16, i32x16, f32);
        spair!(S, f32x4, u32x4, f32);
        spair!(S, f32x8, u32x8, f32);
        spair!(S, f64x2, i64x2, f64);
        spair!(S, f64x4, i64x4, f64);
        spair!(S, f64x8, i64x8, f64);
        spair!(S, f64x2, u64x2, f64);
        spair!(S, f64x4, u64x4, f64);
        spair!(S, f64x8, u64x8, f64);

        // cross-width compositions: narrow via same-width int...
        spair!(S, f32x8, i16x8, f32);
        spair!(S, f32x4, i8x4, f32);
        spair!(S, f64x4, i32x4, f64);
        spair!(S, f64x4, u8x4, f64);
        spair!(S, f64x16, i16x16, f64);
        // ...and the f32 -> 64-bit widen-then-saturate arm
        spair!(S, f32x4, u64x4, f32);
        spair!(S, f32x8, i64x8, f32);
    }

    // Bound smoke: `S: Simd` alone must be enough for generic saturating
    // float -> int casts (the pairs are bound on the `Simd` trait's slots),
    // including the reduced x2 width.
    fn sat_generic_bounds<S: Simd>() {
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

    // `cast` f64->u64 is full-range truncating now (previously it routed to
    // the `_limited` magic-add polyfill, which ROUNDS and only covers
    // [0, 2^52), so the fractional and the beyond-2^52 case both regressed to
    // `as` behavior). `fast_cast` keeps the 2-op limited trick.
    fn f64_to_u64_cast_truncates<S: Simd>() {
        // fractional in-range values must truncate like `as`, not round
        cpair!(S, f64x2, u64x2, f64, |x: f64| if x.is_finite() { (x.abs() % 1000.0) + 0.7 } else { 2.7 }, Tol::Exact);
        cpair!(S, f64x4, u64x4, f64, |x: f64| if x.is_finite() { (x.abs() % 1000.0) + 0.7 } else { 2.7 }, Tol::Exact);
        cpair!(S, f64x8, u64x8, f64, |x: f64| if x.is_finite() { (x.abs() % 1000.0) + 0.7 } else { 2.7 }, Tol::Exact);
        // the full in-range domain, including beyond the old 2^52 limit
        cpair!(S, f64x2, u64x2, f64, |x: f64| if x.is_finite() { x.abs().clamp(0.0, 1.8e19) } else { 2.7 }, Tol::Exact);
        cpair!(S, f64x4, u64x4, f64, |x: f64| if x.is_finite() { x.abs().clamp(0.0, 1.8e19) } else { 2.7 }, Tol::Exact);
        cpair!(S, f64x8, u64x8, f64, |x: f64| if x.is_finite() { x.abs().clamp(0.0, 1.8e19) } else { 2.7 }, Tol::Exact);
    }
}

// wasm's `cast` is already total and `as`-exact in both directions
// (`*_trunc_sat_*` for f32, per-lane scalar `as` for f64), so the raw
// corpus (NaN, infinities, out-of-range) must match the scalar oracle.
// Only `fast_cast` (relaxed trunc / `_limited` magic) is narrow-domain.
// x86 returns the hardware "indefinite" integer there, a documented
// divergence, so this stays wasm-only.
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
mod wasm_total_cast {
    use super::*;
    use thermite::backend::wasm::Wasm;

    #[test]
    fn float_to_int_out_of_range() {
        cpair!(Wasm, f64x4, i64x4, f64, |x| x, Tol::Exact);
        cpair!(Wasm, f32x4, i32x4, f32, |x| x, Tol::Exact);
    }
}
