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
//! 10x10 matrix minus the diagonal): every int-source `cast` and
//! `saturating_cast`, every float-source `saturating_cast` over the raw corpus,
//! every float-source `cast` over the in-range domain, `fast_cast` over its
//! documented domain, and float<->float both ways.
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

macro_rules! lbl {
    ($src:ident, $dst:ident $(, $extra:literal)?) => {
        harness::label::<S>(concat!(stringify!($src), "->", stringify!($dst) $(, $extra)?))
    };
}

macro_rules! row_cast {
    ($src:ident, $se:ty, [$($dst:ident),* $(,)?]) => {$(
        cast_diff!(
            lbl!($src, $dst),
            <S as Simd>::$src, <S as Simd>::$dst,
            <Scalar as Simd>::$src, <Scalar as Simd>::$dst,
            $se, id!($se), Tol::Exact
        );
    )*};
}

macro_rules! row_sat {
    ($src:ident, $se:ty, [$($dst:ident),* $(,)?]) => {$(
        sat_cast_diff!(
            lbl!($src, $dst),
            <S as Simd>::$src, <S as Simd>::$dst,
            <Scalar as Simd>::$src, <Scalar as Simd>::$dst,
            $se
        );
    )*};
}

macro_rules! dom {
    ($fe:ty, $lo:expr, $hi:expr) => {
        (|x: $fe| if x.is_finite() { x.clamp($lo, $hi) } else { 0.0 }) as fn($fe) -> $fe
    };
}

macro_rules! fdom {
    ($fe:ty, $lo:expr, $hi:expr) => {
        (|x: $fe| if x.is_finite() { x.clamp($lo, $hi).trunc() } else { 0.0 }) as fn($fe) -> $fe
    };
}

macro_rules! idom {
    ($ie:ty, $lo:expr, $hi:expr) => {
        (|x: $ie| x.clamp($lo, $hi)) as fn($ie) -> $ie
    };
}

#[rustfmt::skip]
macro_rules! fast_dom {
    ($src:ident, $se:ty, $dst:ident, $prep:expr) => {
        fast_cast_diff!(
            lbl!($src, $dst, " [fast, in-domain]"),
            <S as Simd>::$src, <S as Simd>::$dst,
            <Scalar as Simd>::$src, <Scalar as Simd>::$dst,
            $se, $prep
        );
    };
}

#[rustfmt::skip]
macro_rules! cast_dom {
    ($src:ident, $se:ty, $dst:ident, $lo:expr, $hi:expr) => {
        cast_diff!(
            lbl!($src, $dst, " [in-range]"),
            <S as Simd>::$src, <S as Simd>::$dst,
            <Scalar as Simd>::$src, <Scalar as Simd>::$dst,
            $se, dom!($se, $lo, $hi), Tol::Exact
        );
    };
}

// --- the six per-width bodies ----------------------------------------------

#[rustfmt::skip]
macro_rules! int_src_cast {
    ($f32:ident, $f64:ident, $i8:ident, $u8:ident, $i16:ident, $u16:ident, $i32:ident, $u32:ident, $i64:ident, $u64:ident) => {{
        row_cast!($i8,  i8,  [$f32, $f64, $u8, $i16, $u16, $i32, $u32, $i64, $u64]);
        row_cast!($u8,  u8,  [$f32, $f64, $i8, $i16, $u16, $i32, $u32, $i64, $u64]);
        row_cast!($i16, i16, [$f32, $f64, $i8, $u8,  $u16, $i32, $u32, $i64, $u64]);
        row_cast!($u16, u16, [$f32, $f64, $i8, $u8,  $i16, $i32, $u32, $i64, $u64]);
        row_cast!($i32, i32, [$f32, $f64, $i8, $u8,  $i16, $u16, $u32, $i64, $u64]);
        row_cast!($u32, u32, [$f32, $f64, $i8, $u8,  $i16, $u16, $i32, $i64, $u64]);
        row_cast!($i64, i64, [$f32, $f64, $i8, $u8,  $i16, $u16, $i32, $u32, $u64]);
        row_cast!($u64, u64, [$f32, $f64, $i8, $u8,  $i16, $u16, $i32, $u32, $i64]);
    }};
}

#[rustfmt::skip]
macro_rules! int_src_saturating {
    ($f32:ident, $f64:ident, $i8:ident, $u8:ident, $i16:ident, $u16:ident, $i32:ident, $u32:ident, $i64:ident, $u64:ident) => {{
        row_sat!($i8,  i8,  [$f32, $f64, $u8, $i16, $u16, $i32, $u32, $i64, $u64]);
        row_sat!($u8,  u8,  [$f32, $f64, $i8, $i16, $u16, $i32, $u32, $i64, $u64]);
        row_sat!($i16, i16, [$f32, $f64, $i8, $u8,  $u16, $i32, $u32, $i64, $u64]);
        row_sat!($u16, u16, [$f32, $f64, $i8, $u8,  $i16, $i32, $u32, $i64, $u64]);
        row_sat!($i32, i32, [$f32, $f64, $i8, $u8,  $i16, $u16, $u32, $i64, $u64]);
        row_sat!($u32, u32, [$f32, $f64, $i8, $u8,  $i16, $u16, $i32, $i64, $u64]);
        row_sat!($i64, i64, [$f32, $f64, $i8, $u8,  $i16, $u16, $i32, $u32, $u64]);
        row_sat!($u64, u64, [$f32, $f64, $i8, $u8,  $i16, $u16, $i32, $u32, $i64]);
    }};
}

#[rustfmt::skip]
macro_rules! float_src_saturating {
    ($f32:ident, $f64:ident, $i8:ident, $u8:ident, $i16:ident, $u16:ident, $i32:ident, $u32:ident, $i64:ident, $u64:ident) => {{
        row_sat!($f32, f32, [$i8, $u8, $i16, $u16, $i32, $u32, $i64, $u64]);
        row_sat!($f64, f64, [$i8, $u8, $i16, $u16, $i32, $u32, $i64, $u64]);
    }};
}

#[rustfmt::skip]
macro_rules! float_src_cast_in_range {
    ($f32:ident, $f64:ident, $i8:ident, $u8:ident, $i16:ident, $u16:ident, $i32:ident, $u32:ident, $i64:ident, $u64:ident) => {{
        cast_dom!($f32, f32, $i8,  -128.0, 127.0);
        cast_dom!($f32, f32, $u8,  0.0, 255.0);
        cast_dom!($f32, f32, $i16, -32768.0, 32767.0);
        cast_dom!($f32, f32, $u16, 0.0, 65535.0);
        cast_dom!($f32, f32, $i32, -2.0e9, 2.0e9);
        cast_dom!($f32, f32, $u32, 0.0, 4.0e9);
        cast_dom!($f32, f32, $i64, -9.0e18, 9.0e18);
        cast_dom!($f32, f32, $u64, 0.0, 1.8e19);

        cast_dom!($f64, f64, $i8,  -128.0, 127.0);
        cast_dom!($f64, f64, $u8,  0.0, 255.0);
        cast_dom!($f64, f64, $i16, -32768.0, 32767.0);
        cast_dom!($f64, f64, $u16, 0.0, 65535.0);
        cast_dom!($f64, f64, $i32, -2.0e9, 2.0e9);
        cast_dom!($f64, f64, $u32, 0.0, 4.0e9);
        cast_dom!($f64, f64, $i64, -9.0e18, 9.0e18);
        cast_dom!($f64, f64, $u64, 0.0, 1.8e19);
    }};
}

#[rustfmt::skip]
macro_rules! fast_cast_in_domain {
    ($f32:ident, $f64:ident, $i8:ident, $u8:ident, $i16:ident, $u16:ident, $i32:ident, $u32:ident, $i64:ident, $u64:ident) => {{
        const I64_LO: i64 = -(1 << 51);
        const I64_HI: i64 = 1 << 51;
        const U64_HI: u64 = (1 << 52) - 1;

        fast_dom!($f64, f64, $i64, fdom!(f64, -2251799813685248.0, 2251799813685248.0));
        fast_dom!($f64, f64, $u64, fdom!(f64, 0.0, 4503599627370495.0));
        fast_dom!($i64, i64, $f64, idom!(i64, I64_LO, I64_HI));
        fast_dom!($u64, u64, $f64, idom!(u64, 0, U64_HI));

        fast_dom!($f32, f32, $i32, fdom!(f32, -2.0e9, 2.0e9));
        fast_dom!($f32, f32, $u32, fdom!(f32, 0.0, 4.0e9));
        fast_dom!($i32, i32, $f32, idom!(i32, i32::MIN, i32::MAX));
        fast_dom!($u32, u32, $f32, idom!(u32, 0, u32::MAX));
    }};
}

#[rustfmt::skip]
macro_rules! float_to_float {
    ($f32:ident, $f64:ident, $i8:ident, $u8:ident, $i16:ident, $u16:ident, $i32:ident, $u32:ident, $i64:ident, $u64:ident) => {{
        row_cast!($f32, f32, [$f64]);
        row_cast!($f64, f64, [$f32]);
        row_sat!($f32, f32, [$f64]);
        row_sat!($f64, f64, [$f32]);
    }};
}

macro_rules! x2 {
    ($m:ident) => {
        $m!(f32x2, f64x2, i8x2, u8x2, i16x2, u16x2, i32x2, u32x2, i64x2, u64x2)
    };
}
macro_rules! x4 {
    ($m:ident) => {
        $m!(f32x4, f64x4, i8x4, u8x4, i16x4, u16x4, i32x4, u32x4, i64x4, u64x4)
    };
}
macro_rules! x8 {
    ($m:ident) => {
        $m!(f32x8, f64x8, i8x8, u8x8, i16x8, u16x8, i32x8, u32x8, i64x8, u64x8)
    };
}
macro_rules! x16 {
    ($m:ident) => {
        $m!(
            f32x16, f64x16, i8x16, u8x16, i16x16, u16x16, i32x16, u32x16, i64x16, u64x16
        )
    };
}

for_each_backend_concrete! {
    fn x2_int_src_cast() { x2!(int_src_cast) }
    fn x2_int_src_saturating() { x2!(int_src_saturating) }
    fn x2_float_src_saturating() { x2!(float_src_saturating) }
    fn x2_float_src_cast_in_range() { x2!(float_src_cast_in_range) }
    fn x2_fast_cast_in_domain() { x2!(fast_cast_in_domain) }
    fn x2_float_to_float() { x2!(float_to_float) }

    fn x4_int_src_cast() { x4!(int_src_cast) }
    fn x4_int_src_saturating() { x4!(int_src_saturating) }
    fn x4_float_src_saturating() { x4!(float_src_saturating) }
    fn x4_float_src_cast_in_range() { x4!(float_src_cast_in_range) }
    fn x4_fast_cast_in_domain() { x4!(fast_cast_in_domain) }
    fn x4_float_to_float() { x4!(float_to_float) }

    fn x8_int_src_cast() { x8!(int_src_cast) }
    fn x8_int_src_saturating() { x8!(int_src_saturating) }
    fn x8_float_src_saturating() { x8!(float_src_saturating) }
    fn x8_float_src_cast_in_range() { x8!(float_src_cast_in_range) }
    fn x8_fast_cast_in_domain() { x8!(fast_cast_in_domain) }
    fn x8_float_to_float() { x8!(float_to_float) }

    fn x16_int_src_cast() { x16!(int_src_cast) }
    fn x16_int_src_saturating() { x16!(int_src_saturating) }
    fn x16_float_src_saturating() { x16!(float_src_saturating) }
    fn x16_float_src_cast_in_range() { x16!(float_src_cast_in_range) }
    fn x16_fast_cast_in_domain() { x16!(fast_cast_in_domain) }
    fn x16_float_to_float() { x16!(float_to_float) }
}
