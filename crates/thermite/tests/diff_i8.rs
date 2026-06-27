//! Differential tests for the sub-native 8-bit integer ladder (`Simd` i8x2/x4/x8):
//! every backend register op vs. the `Scalar` reference, plus the i8<->i32 / u8<->u32 widen and
//! narrow casts that the ladder adds. Mirrors `diff_i16.rs`, one element size down.
#![cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32"))]

mod harness;

use harness::Tol;

use thermite::register::{
    BitshiftRegister as _, BitwiseRegister as _, IntegerRegister as _, NumericRegister as _,
    SignedIntegerRegister as _, SignedRegister as _,
};
use thermite::simd::{NativeSimd, Simd};

use thermite::backend::scalar::Scalar;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use thermite::backend::x86_v1::X86V1;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use thermite::backend::x86_v2::X86V2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use thermite::backend::x86_v3::X86V3;

macro_rules! diff_shift {
    ($label:expr, $ut:ty, $rf:ty, $method:ident) => {{
        let mut rng = harness::rng();
        type E = <$ut as thermite::register::Register>::Element;
        let lanes = <<$ut as thermite::register::CoreRegister>::Lanes as generic_array::typenum::Unsigned>::USIZE;
        let bits = (core::mem::size_of::<E>() * 8) as u32;
        for input in harness::corpus::<E>(lanes, &mut rng) {
            for sh in 0..bits {
                let got = harness::read::<$ut>(&<$ut>::$method(harness::make_array::<$ut>(&input), sh));
                let want = harness::read::<$rf>(&<$rf>::$method(harness::make_array::<$rf>(&input), sh));
                harness::assert_lanes_eq(
                    concat!($label, " [", stringify!($method), "]"),
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

macro_rules! int8_tests {
    ($modname:ident, $backend:ty, $reg:ident, $label:expr, signed) => {
        #[test]
        fn $modname() {
            type UT = <$backend as Simd>::$reg;
            type RF = <Scalar as Simd>::$reg;
            int8_common!(UT, RF, $label);
            diff_unary!($label, UT, RF, neg, Tol::Exact);
            diff_unary!($label, UT, RF, abs, Tol::Exact);
            diff_shift!($label, UT, RF, sra);
        }
    };
    ($modname:ident, $backend:ty, $reg:ident, $label:expr, unsigned) => {
        #[test]
        fn $modname() {
            type UT = <$backend as Simd>::$reg;
            type RF = <Scalar as Simd>::$reg;
            int8_common!(UT, RF, $label);
        }
    };
}

macro_rules! backend_suite {
    ($mod:ident, $backend:ty, $label:literal) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        mod $mod {
            use super::*;
            int8_tests!(i8x2, $backend, i8x2, concat!($label, " i8x2"), signed);
            int8_tests!(i8x4, $backend, i8x4, concat!($label, " i8x4"), signed);
            int8_tests!(i8x8, $backend, i8x8, concat!($label, " i8x8"), signed);
            int8_tests!(i8x16, $backend, i8x16, concat!($label, " i8x16"), signed);
            int8_tests!(u8x2, $backend, u8x2, concat!($label, " u8x2"), unsigned);
            int8_tests!(u8x4, $backend, u8x4, concat!($label, " u8x4"), unsigned);
            int8_tests!(u8x8, $backend, u8x8, concat!($label, " u8x8"), unsigned);
            int8_tests!(u8x16, $backend, u8x16, concat!($label, " u8x16"), unsigned);
        }
    };
}

backend_suite!(v1, X86V1, "x86_v1");
backend_suite!(v2, X86V2, "x86_v2");
backend_suite!(v3, X86V3, "x86_v3");

// --- 8<->32 widen/narrow casts (the new `i8<->i32`/`u8<->u32`) vs scalar `as` ---
macro_rules! cast_suite {
    ($mod:ident, $backend:ty, $label:literal) => {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        mod $mod {
            use super::*;
            use thermite::simd::Simd;

            macro_rules! cast8 {
                ($name:ident, $w8:ident, $w32:ident, $se8:ty, $se32:ty) => {
                    #[test]
                    fn $name() {
                        // widen 8 -> 32 (value-preserving)
                        cast_diff!(
                            concat!($label, " ", stringify!($w8), "->", stringify!($w32)),
                            <$backend as Simd>::$w8,
                            <$backend as Simd>::$w32,
                            <Scalar as Simd>::$w8,
                            <Scalar as Simd>::$w32,
                            $se8,
                            |x| x,
                            Tol::Exact
                        );
                        // narrow 32 -> 8 (truncate low 8 bits, like `as`)
                        cast_diff!(
                            concat!($label, " ", stringify!($w32), "->", stringify!($w8)),
                            <$backend as Simd>::$w32,
                            <$backend as Simd>::$w8,
                            <Scalar as Simd>::$w32,
                            <Scalar as Simd>::$w8,
                            $se32,
                            |x| x,
                            Tol::Exact
                        );
                    }
                };
            }

            cast8!(i8_i32_x2, i8x2, i32x2, i8, i32);
            cast8!(i8_i32_x4, i8x4, i32x4, i8, i32);
            cast8!(i8_i32_x8, i8x8, i32x8, i8, i32);
            cast8!(i8_i32_x16, i8x16, i32x16, i8, i32);
            cast8!(u8_u32_x2, u8x2, u32x2, u8, u32);
            cast8!(u8_u32_x4, u8x4, u32x4, u8, u32);
            cast8!(u8_u32_x8, u8x8, u32x8, u8, u32);
            cast8!(u8_u32_x16, u8x16, u32x16, u8, u32);

            // 8<->16 (cast8! is generic over the wider width/elem; reuse it for the 16-bit side)
            cast8!(i8_i16_x2, i8x2, i16x2, i8, i16);
            cast8!(i8_i16_x4, i8x4, i16x4, i8, i16);
            cast8!(i8_i16_x8, i8x8, i16x8, i8, i16);
            cast8!(i8_i16_x16, i8x16, i16x16, i8, i16);
            cast8!(u8_u16_x2, u8x2, u16x2, u8, u16);
            cast8!(u8_u16_x4, u8x4, u16x4, u8, u16);
            cast8!(u8_u16_x8, u8x8, u16x8, u8, u16);
            cast8!(u8_u16_x16, u8x16, u16x16, u8, u16);

            // 8<->64 (reuse cast8! with i64/u64 as the wide side)
            cast8!(i8_i64_x2, i8x2, i64x2, i8, i64);
            cast8!(i8_i64_x4, i8x4, i64x4, i8, i64);
            cast8!(i8_i64_x8, i8x8, i64x8, i8, i64);
            cast8!(i8_i64_x16, i8x16, i64x16, i8, i64);
            cast8!(u8_u64_x2, u8x2, u64x2, u8, u64);
            cast8!(u8_u64_x4, u8x4, u64x4, u8, u64);
            cast8!(u8_u64_x8, u8x8, u64x8, u8, u64);
            cast8!(u8_u64_x16, u8x16, u64x16, u8, u64);
        }
    };
}

cast_suite!(v1_cast, X86V1, "x86_v1");
cast_suite!(v2_cast, X86V2, "x86_v2");
cast_suite!(v3_cast, X86V3, "x86_v3");

// --- WASM ---
#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use thermite::backend::wasm::Wasm;

    int8_tests!(i8x2, Wasm, i8x2, "wasm i8x2", signed);
    int8_tests!(i8x4, Wasm, i8x4, "wasm i8x4", signed);
    int8_tests!(i8x8, Wasm, i8x8, "wasm i8x8", signed);
    int8_tests!(i8x16, Wasm, i8x16, "wasm i8x16", signed);
    int8_tests!(u8x2, Wasm, u8x2, "wasm u8x2", unsigned);
    int8_tests!(u8x4, Wasm, u8x4, "wasm u8x4", unsigned);
    int8_tests!(u8x8, Wasm, u8x8, "wasm u8x8", unsigned);
    int8_tests!(u8x16, Wasm, u8x16, "wasm u8x16", unsigned);
}

#[cfg(target_arch = "wasm32")]
mod wasm_cast {
    use super::*;
    use thermite::backend::wasm::Wasm;
    use thermite::simd::Simd;

    macro_rules! cast8 {
        ($name:ident, $w8:ident, $w32:ident, $se8:ty, $se32:ty) => {
            #[test]
            fn $name() {
                cast_diff!(
                    concat!("wasm ", stringify!($w8), "->", stringify!($w32)),
                    <Wasm as Simd>::$w8,
                    <Wasm as Simd>::$w32,
                    <Scalar as Simd>::$w8,
                    <Scalar as Simd>::$w32,
                    $se8,
                    |x| x,
                    Tol::Exact
                );
                cast_diff!(
                    concat!("wasm ", stringify!($w32), "->", stringify!($w8)),
                    <Wasm as Simd>::$w32,
                    <Wasm as Simd>::$w8,
                    <Scalar as Simd>::$w32,
                    <Scalar as Simd>::$w8,
                    $se32,
                    |x| x,
                    Tol::Exact
                );
            }
        };
    }

    cast8!(i8_i32_x2, i8x2, i32x2, i8, i32);
    cast8!(i8_i32_x4, i8x4, i32x4, i8, i32);
    cast8!(i8_i32_x8, i8x8, i32x8, i8, i32);
    cast8!(i8_i32_x16, i8x16, i32x16, i8, i32);
    cast8!(u8_u32_x2, u8x2, u32x2, u8, u32);
    cast8!(u8_u32_x4, u8x4, u32x4, u8, u32);
    cast8!(u8_u32_x8, u8x8, u32x8, u8, u32);
    cast8!(u8_u32_x16, u8x16, u32x16, u8, u32);

    cast8!(i8_i16_x2, i8x2, i16x2, i8, i16);
    cast8!(i8_i16_x4, i8x4, i16x4, i8, i16);
    cast8!(i8_i16_x8, i8x8, i16x8, i8, i16);
    cast8!(i8_i16_x16, i8x16, i16x16, i8, i16);
    cast8!(u8_u16_x2, u8x2, u16x2, u8, u16);
    cast8!(u8_u16_x4, u8x4, u16x4, u8, u16);
    cast8!(u8_u16_x8, u8x8, u16x8, u8, u16);
    cast8!(u8_u16_x16, u8x16, u16x16, u8, u16);

    cast8!(i8_i64_x2, i8x2, i64x2, i8, i64);
    cast8!(i8_i64_x4, i8x4, i64x4, i8, i64);
    cast8!(i8_i64_x8, i8x8, i64x8, i8, i64);
    cast8!(i8_i64_x16, i8x16, i64x16, i8, i64);
    cast8!(u8_u64_x2, u8x2, u64x2, u8, u64);
    cast8!(u8_u64_x4, u8x4, u64x4, u8, u64);
    cast8!(u8_u64_x8, u8x8, u64x8, u8, u64);
    cast8!(u8_u64_x16, u8x16, u64x16, u8, u64);
}
