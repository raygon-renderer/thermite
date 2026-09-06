//! Differential tests for the fixed-width 8-bit integer ladder (`Simd` i8x2/x4/x8/x16):
//! every backend register op vs. the `Scalar` reference, plus the 8<->16/32/64 widen and
//! narrow casts that the ladder adds. Mirrors `diff_i16.rs`, one element size down.
//! The native-width 8-bit slots (`i8xN`/`u8xN`) are covered in `diff_u8.rs`.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use harness::Tol;

use thermite::register::{
    BitshiftRegister as _, BitwiseRegister as _, IntegerRegister as _, NumericRegister as _,
    SignedIntegerRegister as _, SignedRegister as _,
};
use thermite::simd::Simd;

use thermite::backend::scalar::Scalar;

macro_rules! diff_shift {
    ($label:expr, $ut:ty, $rf:ty, $method:ident) => {{
        let mut rng = harness::rng();
        let lanes = <<$ut as thermite::register::CoreRegister>::Lanes as generic_array::typenum::Unsigned>::USIZE;
        let bits = (core::mem::size_of::<<$ut as thermite::register::Register>::Element>() * 8) as u32;
        for input in harness::corpus::<<$ut as thermite::register::Register>::Element>(lanes, &mut rng) {
            for sh in 0..bits {
                let got = harness::read::<$ut>(&<$ut>::$method(harness::make_array::<$ut>(&input), sh));
                let want = harness::read::<$rf>(&<$rf>::$method(harness::make_array::<$rf>(&input), sh));
                harness::assert_lanes_eq(
                    &format!("{} [{}]", $label, stringify!($method)),
                    &[input.as_slice()],
                    &got,
                    &want,
                    Tol::Exact,
                );
            }
        }
    }};
}

// Common op set shared by signed and unsigned 8-bit registers.
macro_rules! int8_common {
    ($ut:ty, $rf:ty, $label:expr) => {{
        diff_binary!($label, $ut, $rf, add, Tol::Exact);
        diff_binary!($label, $ut, $rf, sub, Tol::Exact);
        diff_binary!($label, $ut, $rf, mul, Tol::Exact);
        diff_binary!($label, $ut, $rf, min, Tol::Exact);
        diff_binary!($label, $ut, $rf, max, Tol::Exact);
        diff_binary!($label, $ut, $rf, saturating_add, Tol::Exact);
        diff_binary!($label, $ut, $rf, saturating_sub, Tol::Exact);
        diff_binary!($label, $ut, $rf, bitand, Tol::Exact);
        diff_binary!($label, $ut, $rf, bitor, Tol::Exact);
        diff_binary!($label, $ut, $rf, bitxor, Tol::Exact);
        diff_binary!($label, $ut, $rf, bitandnot, Tol::Exact);
        diff_unary!($label, $ut, $rf, not, Tol::Exact);
        diff_unary!($label, $ut, $rf, count_ones, Tol::Exact);
        diff_unary!($label, $ut, $rf, leading_zeros, Tol::Exact);
        diff_unary!($label, $ut, $rf, trailing_zeros, Tol::Exact);
        diff_shift!($label, $ut, $rf, shl);
        diff_shift!($label, $ut, $rf, shr);
        diff_reduce!($label, $ut, $rf, min_element, Tol::Exact);
        diff_reduce!($label, $ut, $rf, max_element, Tol::Exact);
    }};
}

macro_rules! int8_ops {
    ($S:ty, $reg:ident, $sign:ident) => {{
        let label = harness::label::<$S>(stringify!($reg));
        int8_ops!(@body <$S as Simd>::$reg, <Scalar as Simd>::$reg, label.as_str(), $sign);
    }};
    (@body $ut:ty, $rf:ty, $label:expr, signed) => {{
        int8_common!($ut, $rf, $label);
        diff_unary!($label, $ut, $rf, neg, Tol::Exact);
        diff_unary!($label, $ut, $rf, abs, Tol::Exact);
        diff_shift!($label, $ut, $rf, sra);
    }};
    (@body $ut:ty, $rf:ty, $label:expr, unsigned) => {{
        int8_common!($ut, $rf, $label);
    }};
}

// 8<->wider widen (value-preserving) and narrow (truncate low 8 bits, like `as`) casts.
macro_rules! cast8 {
    ($S:ty, $w8:ident, $wide:ident, $se8:ty, $sewide:ty) => {{
        cast_diff!(
            harness::label::<$S>(concat!(stringify!($w8), "->", stringify!($wide))),
            <$S as Simd>::$w8,
            <$S as Simd>::$wide,
            <Scalar as Simd>::$w8,
            <Scalar as Simd>::$wide,
            $se8,
            |x| x,
            Tol::Exact
        );
        cast_diff!(
            harness::label::<$S>(concat!(stringify!($wide), "->", stringify!($w8))),
            <$S as Simd>::$wide,
            <$S as Simd>::$w8,
            <Scalar as Simd>::$wide,
            <Scalar as Simd>::$w8,
            $sewide,
            |x| x,
            Tol::Exact
        );
    }};
}

for_each_backend! {
    fn i8x2<S: Simd>() { int8_ops!(S, i8x2, signed) }
    fn i8x4<S: Simd>() { int8_ops!(S, i8x4, signed) }
    fn i8x8<S: Simd>() { int8_ops!(S, i8x8, signed) }
    fn i8x16<S: Simd>() { int8_ops!(S, i8x16, signed) }
    fn u8x2<S: Simd>() { int8_ops!(S, u8x2, unsigned) }
    fn u8x4<S: Simd>() { int8_ops!(S, u8x4, unsigned) }
    fn u8x8<S: Simd>() { int8_ops!(S, u8x8, unsigned) }
    fn u8x16<S: Simd>() { int8_ops!(S, u8x16, unsigned) }

    fn cast_i8_i16<S: Simd>() {
        cast8!(S, i8x2, i16x2, i8, i16);
        cast8!(S, i8x4, i16x4, i8, i16);
        cast8!(S, i8x8, i16x8, i8, i16);
        cast8!(S, i8x16, i16x16, i8, i16);
    }
    fn cast_u8_u16<S: Simd>() {
        cast8!(S, u8x2, u16x2, u8, u16);
        cast8!(S, u8x4, u16x4, u8, u16);
        cast8!(S, u8x8, u16x8, u8, u16);
        cast8!(S, u8x16, u16x16, u8, u16);
    }
    fn cast_i8_i32<S: Simd>() {
        cast8!(S, i8x2, i32x2, i8, i32);
        cast8!(S, i8x4, i32x4, i8, i32);
        cast8!(S, i8x8, i32x8, i8, i32);
        cast8!(S, i8x16, i32x16, i8, i32);
    }
    fn cast_u8_u32<S: Simd>() {
        cast8!(S, u8x2, u32x2, u8, u32);
        cast8!(S, u8x4, u32x4, u8, u32);
        cast8!(S, u8x8, u32x8, u8, u32);
        cast8!(S, u8x16, u32x16, u8, u32);
    }
    fn cast_i8_i64<S: Simd>() {
        cast8!(S, i8x2, i64x2, i8, i64);
        cast8!(S, i8x4, i64x4, i8, i64);
        cast8!(S, i8x8, i64x8, i8, i64);
        cast8!(S, i8x16, i64x16, i8, i64);
    }
    fn cast_u8_u64<S: Simd>() {
        cast8!(S, u8x2, u64x2, u8, u64);
        cast8!(S, u8x4, u64x4, u8, u64);
        cast8!(S, u8x8, u64x8, u8, u64);
        cast8!(S, u8x16, u64x16, u8, u64);
    }
}
