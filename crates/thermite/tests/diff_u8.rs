//! Differential tests for the native-width 8-bit integer families (`SimdExperimental`):
//! every backend register op vs. an element-wise scalar oracle, for i8/u8.
//!
//! The 8-bit families exist only at native width (`i8xN`/`u8xN`) - there is no fixed-width
//! ladder. The scalar backend's native 8-bit slot is 1-lane, so the differential reference is
//! an `ArrayRegister<{i8,u8}, N>` (N = the backend's native byte width: 16 on SSE, 32 on
//! AVX2), which is a pure element-wise scalar register of matching lane count.
#![cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32"))]

mod harness;

use harness::Tol;

use thermite::register::array::ArrayRegister;
use thermite::register::{
    BitshiftRegister as _, BitwiseRegister as _, IntegerRegister as _, NumericRegister as _,
    SignedIntegerRegister as _, SignedRegister as _,
};
use thermite::simd::SimdExperimental;

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

// Common op set shared by signed and unsigned 8-bit registers.
macro_rules! int8_common {
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

macro_rules! int8_tests {
    ($modname:ident, $ut:ty, $rf:ty, $label:expr, signed) => {
        #[test]
        fn $modname() {
            int8_common!($ut, $rf, $label);
            diff_unary!($label, $ut, $rf, neg, Tol::Exact);
            diff_unary!($label, $ut, $rf, abs, Tol::Exact);
            diff_shift!($label, $ut, $rf, sra); // arithmetic (sign-extending) shift
        }
    };
    ($modname:ident, $ut:ty, $rf:ty, $label:expr, unsigned) => {
        #[test]
        fn $modname() {
            int8_common!($ut, $rf, $label);
        }
    };
}

// --- X86V2 (SSE4.2): native 16-lane i8x16/u8x16 ---------------------------
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod v2 {
    use super::*;

    int8_tests!(i8x16, <X86V2 as SimdExperimental>::i8xN, ArrayRegister<i8, 16>, "x86_v2 i8x16", signed);
    int8_tests!(u8x16, <X86V2 as SimdExperimental>::u8xN, ArrayRegister<u8, 16>, "x86_v2 u8x16", unsigned);
}

// --- X86V1 (SSE2): native 16-lane; many ops use SSE2 polyfills / scalar fallbacks ---
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod v1 {
    use super::*;

    int8_tests!(i8x16, <X86V1 as SimdExperimental>::i8xN, ArrayRegister<i8, 16>, "x86_v1 i8x16", signed);
    int8_tests!(u8x16, <X86V1 as SimdExperimental>::u8xN, ArrayRegister<u8, 16>, "x86_v1 u8x16", unsigned);
}

// --- X86V3 (AVX2): native 32-lane i8x32/u8x32 (256-bit) + the fixed 128-bit i8x16 half ---
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod v3 {
    use super::*;

    int8_tests!(i8x32, <X86V3 as SimdExperimental>::i8xN, ArrayRegister<i8, 32>, "x86_v3 i8x32", signed);
    int8_tests!(u8x32, <X86V3 as SimdExperimental>::u8xN, ArrayRegister<u8, 32>, "x86_v3 u8x32", unsigned);

    // The fixed 128-bit i8x16/u8x16 slot (its own register on v3, distinct from the 256-bit native).
    int8_tests!(i8x16, <X86V3 as SimdExperimental>::i8x16, ArrayRegister<i8, 16>, "x86_v3 i8x16", signed);
    int8_tests!(u8x16, <X86V3 as SimdExperimental>::u8x16, ArrayRegister<u8, 16>, "x86_v3 u8x16", unsigned);
}

// --- WASM (SIMD128): native 16-lane i8x16/u8x16 (= the fixed i8x16 slot too) ---
#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use thermite::backend::wasm::Wasm;

    int8_tests!(i8x16, <Wasm as SimdExperimental>::i8xN, ArrayRegister<i8, 16>, "wasm i8x16", signed);
    int8_tests!(u8x16, <Wasm as SimdExperimental>::u8xN, ArrayRegister<u8, 16>, "wasm u8x16", unsigned);
}
