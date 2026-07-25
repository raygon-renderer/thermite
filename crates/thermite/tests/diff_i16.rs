//! Differential tests for the 16-bit integer families (`Simd`): every backend
//! register op vs. the `Scalar` reference, across the i16/u16 width matrix.
//!
//! Only built where the x86 SIMD backends exist. The 16-bit slots live on
//! `Simd` (a staging trait), so register types are resolved through it rather
//! than `Simd`.
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
use thermite::simd::{NativeSimd, Simd};

use thermite::backend::scalar::Scalar;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use thermite::backend::x86_v1::X86V1;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use thermite::backend::x86_v2::X86V2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use thermite::backend::x86_v3::X86V3;

// Uniform scalar-amount shift stamper (the harness only exports the vector-input variants).
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
        diff_reduce!($label, $ut, $rf, sum_elements, Tol::Exact);
        diff_reduce!($label, $ut, $rf, prod_elements, Tol::Exact);
        diff_reduce!($label, $ut, $rf, min_element, Tol::Exact);
        diff_reduce!($label, $ut, $rf, max_element, Tol::Exact);
    }};
}

macro_rules! int16_tests {
    ($modname:ident, $backend:ty, $reg:ident, $label:expr, signed) => {
        #[test]
        fn $modname() {
            type UT = <$backend as Simd>::$reg;
            type RF = <Scalar as Simd>::$reg;
            int16_common!(UT, RF, $label);
            diff_unary!($label, UT, RF, neg, Tol::Exact);
            diff_unary!($label, UT, RF, abs, Tol::Exact);
            diff_binary!($label, UT, RF, mulhrs, Tol::Exact); // native PMULHRSW vs scalar polyfill
            diff_shift!($label, UT, RF, sra); // arithmetic (sign-extending) shift
        }
    };
    ($modname:ident, $backend:ty, $reg:ident, $label:expr, unsigned) => {
        #[test]
        fn $modname() {
            type UT = <$backend as Simd>::$reg;
            type RF = <Scalar as Simd>::$reg;
            int16_common!(UT, RF, $label);
        }
    };
}

// --- X86V2 (SSE4.2) vs Scalar ---------------------------------------------
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod v2 {
    use super::*;

    int16_tests!(i16x2, X86V2, i16x2, "x86_v2 i16x2", signed);
    int16_tests!(i16x4, X86V2, i16x4, "x86_v2 i16x4", signed);
    int16_tests!(i16x8, X86V2, i16x8, "x86_v2 i16x8", signed);
    int16_tests!(i16x16, X86V2, i16x16, "x86_v2 i16x16", signed);

    int16_tests!(u16x2, X86V2, u16x2, "x86_v2 u16x2", unsigned);
    int16_tests!(u16x4, X86V2, u16x4, "x86_v2 u16x4", unsigned);
    int16_tests!(u16x8, X86V2, u16x8, "x86_v2 u16x8", unsigned);
    int16_tests!(u16x16, X86V2, u16x16, "x86_v2 u16x16", unsigned);
}

// --- X86V1 (SSE2) vs Scalar. x8 native 128-bit; many ops use SSE2/scalar fallbacks. ---
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod v1 {
    use super::*;

    int16_tests!(i16x2, X86V1, i16x2, "x86_v1 i16x2", signed);
    int16_tests!(i16x4, X86V1, i16x4, "x86_v1 i16x4", signed);
    int16_tests!(i16x8, X86V1, i16x8, "x86_v1 i16x8", signed);
    int16_tests!(i16x16, X86V1, i16x16, "x86_v1 i16x16", signed);

    int16_tests!(u16x2, X86V1, u16x2, "x86_v1 u16x2", unsigned);
    int16_tests!(u16x4, X86V1, u16x4, "x86_v1 u16x4", unsigned);
    int16_tests!(u16x8, X86V1, u16x8, "x86_v1 u16x8", unsigned);
    int16_tests!(u16x16, X86V1, u16x16, "x86_v1 u16x16", unsigned);
}

// --- X86V3 (AVX2) vs Scalar. x8 is native 128-bit, x16 is native 256-bit. ---
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod v3 {
    use super::*;

    int16_tests!(i16x2, X86V3, i16x2, "x86_v3 i16x2", signed);
    int16_tests!(i16x4, X86V3, i16x4, "x86_v3 i16x4", signed);
    int16_tests!(i16x8, X86V3, i16x8, "x86_v3 i16x8", signed);
    int16_tests!(i16x16, X86V3, i16x16, "x86_v3 i16x16", signed);

    int16_tests!(u16x2, X86V3, u16x2, "x86_v3 u16x2", unsigned);
    int16_tests!(u16x4, X86V3, u16x4, "x86_v3 u16x4", unsigned);
    int16_tests!(u16x8, X86V3, u16x8, "x86_v3 u16x8", unsigned);
    int16_tests!(u16x16, X86V3, u16x16, "x86_v3 u16x16", unsigned);
}

// --- 16<->32 widen/narrow casts (the required `i16<->i32`/`u16<->u32`) vs scalar `as` ---
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod v2_cast {
    use super::*;
    use thermite::simd::Simd;

    // widen i16->i32 / u16->u32 (value-preserving), and narrow i32->i16 / u32->u16
    // (truncating, like `as`). One #[test] per width.
    macro_rules! cast16 {
        ($name:ident, $w16:ident, $w32:ident, $se16:ty, $se32:ty) => {
            #[test]
            fn $name() {
                // widen 16 -> 32
                cast_diff!(
                    concat!("x86_v2 ", stringify!($w16), "->", stringify!($w32)),
                    <X86V2 as Simd>::$w16,
                    <X86V2 as Simd>::$w32,
                    <Scalar as Simd>::$w16,
                    <Scalar as Simd>::$w32,
                    $se16,
                    |x| x,
                    Tol::Exact
                );
                // narrow 32 -> 16 (truncate low 16 bits)
                cast_diff!(
                    concat!("x86_v2 ", stringify!($w32), "->", stringify!($w16)),
                    <X86V2 as Simd>::$w32,
                    <X86V2 as Simd>::$w16,
                    <Scalar as Simd>::$w32,
                    <Scalar as Simd>::$w16,
                    $se32,
                    |x| x,
                    Tol::Exact
                );
            }
        };
    }

    cast16!(i16_i32_x2, i16x2, i32x2, i16, i32);
    cast16!(i16_i32_x4, i16x4, i32x4, i16, i32);
    cast16!(i16_i32_x8, i16x8, i32x8, i16, i32);
    cast16!(u16_u32_x2, u16x2, u32x2, u16, u32);
    cast16!(u16_u32_x4, u16x4, u32x4, u16, u32);
    cast16!(u16_u32_x8, u16x8, u32x8, u16, u32);

    // 16<->64 (reuse cast16! with i64/u64 as the wide side)
    cast16!(i16_i64_x2, i16x2, i64x2, i16, i64);
    cast16!(i16_i64_x4, i16x4, i64x4, i16, i64);
    cast16!(i16_i64_x8, i16x8, i64x8, i16, i64);
    cast16!(i16_i64_x16, i16x16, i64x16, i16, i64);
    cast16!(u16_u64_x2, u16x2, u64x2, u16, u64);
    cast16!(u16_u64_x4, u16x4, u64x4, u16, u64);
    cast16!(u16_u64_x8, u16x8, u64x8, u16, u64);
    cast16!(u16_u64_x16, u16x16, u64x16, u16, u64);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod v3_cast {
    use super::*;
    use thermite::simd::Simd;

    macro_rules! cast16 {
        ($name:ident, $w16:ident, $w32:ident, $se16:ty, $se32:ty) => {
            #[test]
            fn $name() {
                cast_diff!(
                    concat!("x86_v3 ", stringify!($w16), "->", stringify!($w32)),
                    <X86V3 as Simd>::$w16,
                    <X86V3 as Simd>::$w32,
                    <Scalar as Simd>::$w16,
                    <Scalar as Simd>::$w32,
                    $se16,
                    |x| x,
                    Tol::Exact
                );
                cast_diff!(
                    concat!("x86_v3 ", stringify!($w32), "->", stringify!($w16)),
                    <X86V3 as Simd>::$w32,
                    <X86V3 as Simd>::$w16,
                    <Scalar as Simd>::$w32,
                    <Scalar as Simd>::$w16,
                    $se32,
                    |x| x,
                    Tol::Exact
                );
            }
        };
    }

    cast16!(i16_i32_x2, i16x2, i32x2, i16, i32);
    cast16!(i16_i32_x4, i16x4, i32x4, i16, i32);
    cast16!(i16_i32_x8, i16x8, i32x8, i16, i32);
    cast16!(i16_i32_x16, i16x16, i32x16, i16, i32);
    cast16!(u16_u32_x2, u16x2, u32x2, u16, u32);
    cast16!(u16_u32_x4, u16x4, u32x4, u16, u32);
    cast16!(u16_u32_x8, u16x8, u32x8, u16, u32);
    cast16!(u16_u32_x16, u16x16, u32x16, u16, u32);

    cast16!(i16_i64_x2, i16x2, i64x2, i16, i64);
    cast16!(i16_i64_x4, i16x4, i64x4, i16, i64);
    cast16!(i16_i64_x8, i16x8, i64x8, i16, i64);
    cast16!(i16_i64_x16, i16x16, i64x16, i16, i64);
    cast16!(u16_u64_x2, u16x2, u64x2, u16, u64);
    cast16!(u16_u64_x4, u16x4, u64x4, u16, u64);
    cast16!(u16_u64_x8, u16x8, u64x8, u16, u64);
    cast16!(u16_u64_x16, u16x16, u64x16, u16, u64);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod v1_cast {
    use super::*;
    use thermite::simd::Simd;

    macro_rules! cast16 {
        ($name:ident, $w16:ident, $w32:ident, $se16:ty, $se32:ty) => {
            #[test]
            fn $name() {
                cast_diff!(
                    concat!("x86_v1 ", stringify!($w16), "->", stringify!($w32)),
                    <X86V1 as Simd>::$w16,
                    <X86V1 as Simd>::$w32,
                    <Scalar as Simd>::$w16,
                    <Scalar as Simd>::$w32,
                    $se16,
                    |x| x,
                    Tol::Exact
                );
                cast_diff!(
                    concat!("x86_v1 ", stringify!($w32), "->", stringify!($w16)),
                    <X86V1 as Simd>::$w32,
                    <X86V1 as Simd>::$w16,
                    <Scalar as Simd>::$w32,
                    <Scalar as Simd>::$w16,
                    $se32,
                    |x| x,
                    Tol::Exact
                );
            }
        };
    }

    cast16!(i16_i32_x2, i16x2, i32x2, i16, i32);
    cast16!(i16_i32_x4, i16x4, i32x4, i16, i32);
    cast16!(i16_i32_x8, i16x8, i32x8, i16, i32);
    cast16!(i16_i32_x16, i16x16, i32x16, i16, i32);
    cast16!(u16_u32_x2, u16x2, u32x2, u16, u32);
    cast16!(u16_u32_x4, u16x4, u32x4, u16, u32);
    cast16!(u16_u32_x8, u16x8, u32x8, u16, u32);
    cast16!(u16_u32_x16, u16x16, u32x16, u16, u32);

    cast16!(i16_i64_x2, i16x2, i64x2, i16, i64);
    cast16!(i16_i64_x4, i16x4, i64x4, i16, i64);
    cast16!(i16_i64_x8, i16x8, i64x8, i16, i64);
    cast16!(i16_i64_x16, i16x16, i64x16, i16, i64);
    cast16!(u16_u64_x2, u16x2, u64x2, u16, u64);
    cast16!(u16_u64_x4, u16x4, u64x4, u16, u64);
    cast16!(u16_u64_x8, u16x8, u64x8, u16, u64);
    cast16!(u16_u64_x16, u16x16, u64x16, u16, u64);
}

// --- WASM (SIMD128): native 8-lane i16x8 (= i16xN), reduced i16x4, array i16x2/x16 ---
#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use thermite::backend::wasm::Wasm;

    int16_tests!(i16x2, Wasm, i16x2, "wasm i16x2", signed);
    int16_tests!(i16x4, Wasm, i16x4, "wasm i16x4", signed);
    int16_tests!(i16x8, Wasm, i16x8, "wasm i16x8", signed);
    int16_tests!(i16x16, Wasm, i16x16, "wasm i16x16", signed);

    int16_tests!(u16x2, Wasm, u16x2, "wasm u16x2", unsigned);
    int16_tests!(u16x4, Wasm, u16x4, "wasm u16x4", unsigned);
    int16_tests!(u16x8, Wasm, u16x8, "wasm u16x8", unsigned);
    int16_tests!(u16x16, Wasm, u16x16, "wasm u16x16", unsigned);
}

// --- NEON: native 8-lane i16x8 (= i16xN); sub-native reduced, wider via ArrayRegister ---
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use thermite::backend::neon::Neon;

    int16_tests!(i16x2, Neon, i16x2, "neon i16x2", signed);
    int16_tests!(i16x4, Neon, i16x4, "neon i16x4", signed);
    int16_tests!(i16x8, Neon, i16x8, "neon i16x8", signed);
    int16_tests!(i16x16, Neon, i16x16, "neon i16x16", signed);

    int16_tests!(u16x2, Neon, u16x2, "neon u16x2", unsigned);
    int16_tests!(u16x4, Neon, u16x4, "neon u16x4", unsigned);
    int16_tests!(u16x8, Neon, u16x8, "neon u16x8", unsigned);
    int16_tests!(u16x16, Neon, u16x16, "neon u16x16", unsigned);
}

#[cfg(target_arch = "wasm32")]
mod wasm_cast {
    use super::*;
    use thermite::backend::wasm::Wasm;
    use thermite::simd::Simd;

    macro_rules! cast16 {
        ($name:ident, $w16:ident, $w32:ident, $se16:ty, $se32:ty) => {
            #[test]
            fn $name() {
                cast_diff!(
                    concat!("wasm ", stringify!($w16), "->", stringify!($w32)),
                    <Wasm as Simd>::$w16,
                    <Wasm as Simd>::$w32,
                    <Scalar as Simd>::$w16,
                    <Scalar as Simd>::$w32,
                    $se16,
                    |x| x,
                    Tol::Exact
                );
                cast_diff!(
                    concat!("wasm ", stringify!($w32), "->", stringify!($w16)),
                    <Wasm as Simd>::$w32,
                    <Wasm as Simd>::$w16,
                    <Scalar as Simd>::$w32,
                    <Scalar as Simd>::$w16,
                    $se32,
                    |x| x,
                    Tol::Exact
                );
            }
        };
    }

    cast16!(i16_i32_x2, i16x2, i32x2, i16, i32);
    cast16!(i16_i32_x4, i16x4, i32x4, i16, i32);
    cast16!(i16_i32_x8, i16x8, i32x8, i16, i32);
    cast16!(u16_u32_x2, u16x2, u32x2, u16, u32);
    cast16!(u16_u32_x4, u16x4, u32x4, u16, u32);
    cast16!(u16_u32_x8, u16x8, u32x8, u16, u32);

    cast16!(i16_i64_x2, i16x2, i64x2, i16, i64);
    cast16!(i16_i64_x4, i16x4, i64x4, i16, i64);
    cast16!(i16_i64_x8, i16x8, i64x8, i16, i64);
    cast16!(i16_i64_x16, i16x16, i64x16, i16, i64);
    cast16!(u16_u64_x2, u16x2, u64x2, u16, u64);
    cast16!(u16_u64_x4, u16x4, u64x4, u16, u64);
    cast16!(u16_u64_x8, u16x8, u64x8, u16, u64);
    cast16!(u16_u64_x16, u16x16, u64x16, u16, u64);
}

#[cfg(target_arch = "aarch64")]
mod neon_cast {
    use super::*;
    use thermite::backend::neon::Neon;
    use thermite::simd::Simd;

    macro_rules! cast16 {
        ($name:ident, $w16:ident, $w32:ident, $se16:ty, $se32:ty) => {
            #[test]
            fn $name() {
                cast_diff!(
                    concat!("neon ", stringify!($w16), "->", stringify!($w32)),
                    <Neon as Simd>::$w16,
                    <Neon as Simd>::$w32,
                    <Scalar as Simd>::$w16,
                    <Scalar as Simd>::$w32,
                    $se16,
                    |x| x,
                    Tol::Exact
                );
                cast_diff!(
                    concat!("neon ", stringify!($w32), "->", stringify!($w16)),
                    <Neon as Simd>::$w32,
                    <Neon as Simd>::$w16,
                    <Scalar as Simd>::$w32,
                    <Scalar as Simd>::$w16,
                    $se32,
                    |x| x,
                    Tol::Exact
                );
            }
        };
    }

    cast16!(i16_i32_x2, i16x2, i32x2, i16, i32);
    cast16!(i16_i32_x4, i16x4, i32x4, i16, i32);
    cast16!(i16_i32_x8, i16x8, i32x8, i16, i32);
    cast16!(u16_u32_x2, u16x2, u32x2, u16, u32);
    cast16!(u16_u32_x4, u16x4, u32x4, u16, u32);
    cast16!(u16_u32_x8, u16x8, u32x8, u16, u32);

    cast16!(i16_i64_x2, i16x2, i64x2, i16, i64);
    cast16!(i16_i64_x4, i16x4, i64x4, i16, i64);
    cast16!(i16_i64_x8, i16x8, i64x8, i16, i64);
    cast16!(i16_i64_x16, i16x16, i64x16, i16, i64);
    cast16!(u16_u64_x2, u16x2, u64x2, u16, u64);
    cast16!(u16_u64_x4, u16x4, u64x4, u16, u64);
    cast16!(u16_u64_x8, u16x8, u64x8, u16, u64);
    cast16!(u16_u64_x16, u16x16, u64x16, u16, u64);
}
