//! Differential tests for the 16-bit integer families: every backend register
//! op vs. the `Scalar` reference, across the i16/u16 width matrix, plus the
//! 16<->32 and 16<->64 widen/narrow casts vs scalar `as`.
//!
//! The fixed-width slots (x2..x16) come from `Simd`. The native-width slot
//! (`i16xN`: x8 on 128-bit backends, x16 on AVX2, x32 on AVX-512) comes from
//! `NativeSimd` and is diffed against an `ArrayRegister` of matching width.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use generic_array::typenum::Unsigned;
use harness::Tol;

use thermite::register::array::ArrayRegister;
use thermite::register::{
    BitshiftRegister as _, BitwiseRegister as _, CoreRegister, IntegerRegister as _, NumericRegister as _,
    SignedIntegerRegister as _, SignedRegister as _,
};
use thermite::simd::{NativeSimd, Simd};

use thermite::backend::scalar::Scalar;

// Uniform scalar-amount shift stamper (the harness only exports the vector-input variants).
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

// Common op set shared by signed and unsigned 16-bit registers.
macro_rules! int16_common {
    ($ut:ty, $rf:ty, $label:expr) => {{
        diff_binary!($label, $ut, $rf, add, Tol::Exact);
        diff_binary!($label, $ut, $rf, sub, Tol::Exact);
        diff_binary!($label, $ut, $rf, mul, Tol::Exact);
        diff_binary!($label, $ut, $rf, mulhi, Tol::Exact);
        diff_binary!($label, $ut, $rf, mullo, Tol::Exact);
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
        // x86 has no variable 16-bit shift before AVX-512BW+VL. These exercise
        // the `_mm*_s{ll,rl}v_epi16x_*` polyfills.
        diff_varshift!($label, $ut, $rf, shlv);
        diff_varshift!($label, $ut, $rf, shrv);
        diff_reduce!($label, $ut, $rf, sum_elements, Tol::Exact);
        diff_reduce!($label, $ut, $rf, prod_elements, Tol::Exact);
        diff_reduce!($label, $ut, $rf, min_element, Tol::Exact);
        diff_reduce!($label, $ut, $rf, max_element, Tol::Exact);
    }};
}

macro_rules! int16_ops {
    ($S:ty, $reg:ident, $sign:ident) => {{
        let label = harness::label::<$S>(stringify!($reg));
        int16_ops!(@body <$S as Simd>::$reg, <Scalar as Simd>::$reg, label.as_str(), $sign);
    }};
    (@body $ut:ty, $rf:ty, $label:expr, signed) => {{
        int16_common!($ut, $rf, $label);
        diff_unary!($label, $ut, $rf, neg, Tol::Exact);
        diff_unary!($label, $ut, $rf, abs, Tol::Exact);
        diff_binary!($label, $ut, $rf, mulhrs, Tol::Exact); // native PMULHRSW vs scalar polyfill
        diff_shift!($label, $ut, $rf, sra); // arithmetic (sign-extending) shift
        diff_varshift!($label, $ut, $rf, srav);
    }};
    (@body $ut:ty, $rf:ty, $label:expr, unsigned) => {{
        int16_common!($ut, $rf, $label);
    }};
}

// The native-width slot is only reachable through `NativeSimd`, and its lane
// count is a per-backend type, so the matching scalar oracle is picked at runtime.
macro_rules! native16 {
    ($S:ty, $slot:ident, $e:ty, $sign:ident) => {{
        let label = harness::label::<$S>(stringify!($slot));
        match <<<$S as NativeSimd>::$slot as CoreRegister>::Lanes as Unsigned>::USIZE {
            1 => int16_ops!(@body <$S as NativeSimd>::$slot, ArrayRegister<$e, 1>, label.as_str(), $sign),
            8 => int16_ops!(@body <$S as NativeSimd>::$slot, ArrayRegister<$e, 8>, label.as_str(), $sign),
            16 => int16_ops!(@body <$S as NativeSimd>::$slot, ArrayRegister<$e, 16>, label.as_str(), $sign),
            32 => int16_ops!(@body <$S as NativeSimd>::$slot, ArrayRegister<$e, 32>, label.as_str(), $sign),
            n => panic!("{label}: unexpected native 16-bit width {n}"),
        }
    }};
}

// 16<->wider widen (value-preserving) and narrow (truncating, like `as`) casts.
macro_rules! cast16 {
    ($S:ty, $w16:ident, $wide:ident, $se16:ty, $sewide:ty) => {{
        cast_diff!(
            harness::label::<$S>(concat!(stringify!($w16), "->", stringify!($wide))),
            <$S as Simd>::$w16,
            <$S as Simd>::$wide,
            <Scalar as Simd>::$w16,
            <Scalar as Simd>::$wide,
            $se16,
            |x| x,
            Tol::Exact
        );
        cast_diff!(
            harness::label::<$S>(concat!(stringify!($wide), "->", stringify!($w16))),
            <$S as Simd>::$wide,
            <$S as Simd>::$w16,
            <Scalar as Simd>::$wide,
            <Scalar as Simd>::$w16,
            $sewide,
            |x| x,
            Tol::Exact
        );
    }};
}

for_each_backend! {
    fn i16x2<S: Simd>() { int16_ops!(S, i16x2, signed) }
    fn i16x4<S: Simd>() { int16_ops!(S, i16x4, signed) }
    fn i16x8<S: Simd>() { int16_ops!(S, i16x8, signed) }
    fn i16x16<S: Simd>() { int16_ops!(S, i16x16, signed) }
    fn u16x2<S: Simd>() { int16_ops!(S, u16x2, unsigned) }
    fn u16x4<S: Simd>() { int16_ops!(S, u16x4, unsigned) }
    fn u16x8<S: Simd>() { int16_ops!(S, u16x8, unsigned) }
    fn u16x16<S: Simd>() { int16_ops!(S, u16x16, unsigned) }

    fn native_i16xN<S: Simd>() { native16!(S, i16xN, i16, signed) }
    fn native_u16xN<S: Simd>() { native16!(S, u16xN, u16, unsigned) }

    fn cast_i16_i32<S: Simd>() {
        cast16!(S, i16x2, i32x2, i16, i32);
        cast16!(S, i16x4, i32x4, i16, i32);
        cast16!(S, i16x8, i32x8, i16, i32);
        cast16!(S, i16x16, i32x16, i16, i32);
    }
    fn cast_u16_u32<S: Simd>() {
        cast16!(S, u16x2, u32x2, u16, u32);
        cast16!(S, u16x4, u32x4, u16, u32);
        cast16!(S, u16x8, u32x8, u16, u32);
        cast16!(S, u16x16, u32x16, u16, u32);
    }
    fn cast_i16_i64<S: Simd>() {
        cast16!(S, i16x2, i64x2, i16, i64);
        cast16!(S, i16x4, i64x4, i16, i64);
        cast16!(S, i16x8, i64x8, i16, i64);
        cast16!(S, i16x16, i64x16, i16, i64);
    }
    fn cast_u16_u64<S: Simd>() {
        cast16!(S, u16x2, u64x2, u16, u64);
        cast16!(S, u16x4, u64x4, u16, u64);
        cast16!(S, u16x8, u64x8, u16, u64);
        cast16!(S, u16x16, u64x16, u16, u64);
    }
}
