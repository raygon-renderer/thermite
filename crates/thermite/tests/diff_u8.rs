//! Differential tests for the native-width 8-bit integer families (`NativeSimd`
//! `i8xN`/`u8xN`): every backend register op vs. an element-wise scalar oracle.
//!
//! The native byte width is a per-backend type (16 on SSE/WASM/NEON, 32 on AVX2,
//! 64 on AVX-512), and the scalar backend's own native 8-bit slot is 1-lane, so
//! the reference is an `ArrayRegister<{i8,u8}, N>` of matching lane count,
//! picked at runtime. The fixed-width ladder (x2..x16) is in `diff_i8.rs`.
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
        // x86 has no 8-bit shift at any level, so these are emulated by
        // widening/narrowing, the code shape that saturated instead of
        // truncating in fearless_simd #287/#289.
        diff_varshift!($label, $ut, $rf, shlv);
        diff_varshift!($label, $ut, $rf, shrv);
        diff_reduce!($label, $ut, $rf, sum_elements, Tol::Exact);
        diff_reduce!($label, $ut, $rf, prod_elements, Tol::Exact);
        diff_reduce!($label, $ut, $rf, min_element, Tol::Exact);
        diff_reduce!($label, $ut, $rf, max_element, Tol::Exact);
    }};
}

macro_rules! int8_ops {
    ($ut:ty, $rf:ty, $label:expr, signed) => {{
        int8_common!($ut, $rf, $label);
        diff_unary!($label, $ut, $rf, neg, Tol::Exact);
        diff_unary!($label, $ut, $rf, abs, Tol::Exact);
        diff_shift!($label, $ut, $rf, sra); // arithmetic (sign-extending) shift
        diff_varshift!($label, $ut, $rf, srav);
    }};
    ($ut:ty, $rf:ty, $label:expr, unsigned) => {{
        int8_common!($ut, $rf, $label);
    }};
}

macro_rules! native8 {
    ($S:ty, $slot:ident, $e:ty, $sign:ident) => {{
        let label = harness::label::<$S>(stringify!($slot));
        match <<<$S as NativeSimd>::$slot as CoreRegister>::Lanes as Unsigned>::USIZE {
            1 => int8_ops!(<$S as NativeSimd>::$slot, ArrayRegister<$e, 1>, label.as_str(), $sign),
            16 => int8_ops!(<$S as NativeSimd>::$slot, ArrayRegister<$e, 16>, label.as_str(), $sign),
            32 => int8_ops!(<$S as NativeSimd>::$slot, ArrayRegister<$e, 32>, label.as_str(), $sign),
            64 => int8_ops!(<$S as NativeSimd>::$slot, ArrayRegister<$e, 64>, label.as_str(), $sign),
            n => panic!("{label}: unexpected native 8-bit width {n}"),
        }
    }};
}

for_each_backend! {
    fn native_i8xN<S: Simd>() { native8!(S, i8xN, i8, signed) }
    fn native_u8xN<S: Simd>() { native8!(S, u8xN, u8, unsigned) }
}
