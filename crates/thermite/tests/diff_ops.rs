//! Differential tests: every SIMD backend register op vs. the `Scalar`
//! reference, across the element-type × width matrix.
//!
//! See `harness/mod.rs` for the methodology. Only built where the x86 SIMD
//! backends exist; elsewhere there is nothing to differentiate against.
#![cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32"))]

mod harness;

use harness::Tol;

// Register traits supply the methods the macros call (add, sqrt, bitand, ...).
use thermite::register::{
    BitshiftRegister as _, BitwiseRegister as _, FloatRegister as _, IntegerRegister as _, NumericRegister as _,
    SignedIntegerRegister as _, SignedRegister as _,
};
use thermite::simd::Simd;

// Backend marker types. Scalar is the differential oracle used inside the macros.
use thermite::backend::scalar::Scalar;

// ---------------------------------------------------------------------------
// Shift op needs a scalar shift amount, so it gets its own stamper.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Per-lane variable shift (`shlv`/`shrv`/`srav`): the shift amount is itself a
// vector, one count per lane. On SSE the 64-bit forms have no native
// instruction and are polyfilled - a lane-swap bug in the 64-bit polyfill (used
// by v1 and v2) silently transposed lanes and was previously untested, since
// `diff_shift!` only covers the uniform scalar-amount form. Differential vs the
// scalar backend over per-lane random shift amounts in `[0, bits)`.
// ---------------------------------------------------------------------------
macro_rules! diff_varshift {
    ($label:expr, $ut:ty, $rf:ty, $method:ident) => {{
        use rand::RngExt as _;
        type E = <$ut as thermite::register::Register>::Element;
        type UUT = <$ut as thermite::register::Register>::Unsigned;
        type URF = <$rf as thermite::register::Register>::Unsigned;
        type UE = <UUT as thermite::register::Register>::Element;
        let mut rng = harness::rng();
        let lanes = <<$ut as thermite::register::CoreRegister>::Lanes as generic_array::typenum::Unsigned>::USIZE;
        let bits = (core::mem::size_of::<E>() * 8) as UE;
        for input in harness::corpus::<E>(lanes, &mut rng) {
            // Distinct per-lane shift amounts so a lane transposition is visible.
            let sh: Vec<UE> = (0..lanes).map(|_| rng.random::<UE>() % bits).collect();
            let ut_sh = harness::make_array::<UUT>(&sh);
            let rf_sh = harness::make_array::<URF>(&sh);
            let got = harness::read::<$ut>(&<$ut>::$method(harness::make_array::<$ut>(&input), ut_sh));
            let want = harness::read::<$rf>(&<$rf>::$method(harness::make_array::<$rf>(&input), rf_sh));
            harness::assert_lanes_eq(
                concat!($label, " [", stringify!($method), "]"),
                &[input.as_slice()],
                &got,
                &want,
                Tol::Exact,
            );
        }
    }};
}

// ---------------------------------------------------------------------------
// Float register suite: one #[test] per (backend, width).
// ---------------------------------------------------------------------------
macro_rules! float_reg_tests {
    ($modname:ident, $ut_backend:ty, $reg:ident, $label:expr) => {
        #[test]
        fn $modname() {
            type UT = <$ut_backend as Simd>::$reg;
            type RF = <Scalar as Simd>::$reg;

            // IEEE-correctly-rounded ops: must be bit-exact vs. scalar.
            diff_binary!($label, UT, RF, add, Tol::Exact);
            diff_binary!($label, UT, RF, sub, Tol::Exact);
            diff_binary!($label, UT, RF, mul, Tol::Exact);
            diff_binary!($label, UT, RF, div, Tol::Exact);
            diff_unary!($label, UT, RF, sqrt, Tol::Exact);

            // Sign / ordering: thermite-defined, scalar backend is the oracle.
            diff_unary!($label, UT, RF, neg, Tol::Exact);
            diff_unary!($label, UT, RF, abs, Tol::Exact);
            // Rel(0.0) == exact, except it treats +0.0 and -0.0 as equal
            // (numerically they are; which signed zero min/max returns is
            // unspecified and differs harmlessly between backends).
            diff_binary_finite!($label, UT, RF, min, Tol::Rel(0.0));
            diff_binary_finite!($label, UT, RF, max, Tol::Rel(0.0));
            diff_unary!($label, UT, RF, floor, Tol::Exact);
            diff_unary!($label, UT, RF, ceil, Tol::Exact);
            diff_unary!($label, UT, RF, trunc, Tol::Exact);
            // NOTE: `round` is intentionally excluded - its half-way rounding
            // direction diverges between backends (scalar = half-away-from-zero,
            // x86 = half-to-even). Flagged in TESTING.md.

            // NOTE: `rcp`/`rsqrt` are hardware approximations and flush
            // denormals; they are accuracy-tested as a property (rcp(x)*x ≈ 1)
            // in `approx_recip.rs`, not differentially against exact scalar.

            // Horizontal reductions. `sum_elements` is non-associative so its
            // tree-vs-fold rounding diverges on adversarial inputs - it gets a
            // tame-input accuracy test in `approx_recip.rs` instead. min/max
            // are associative, so they must agree (modulo NaN).
            diff_reduce_finite!($label, UT, RF, min_element, Tol::Rel(0.0));
            diff_reduce_finite!($label, UT, RF, max_element, Tol::Rel(0.0));
        }
    };
}

// ---------------------------------------------------------------------------
// Integer register suite. Every op below is correct on every integer
// width/backend - the two defects the harness originally found here
// (32-bit reductions, 64-bit `mul`) have been fixed; see TESTING.md.
// ---------------------------------------------------------------------------
macro_rules! int_reg_tests {
    ($modname:ident, $ut_backend:ty, $reg:ident, $label:expr, signed) => {
        #[test]
        fn $modname() {
            type UT = <$ut_backend as Simd>::$reg;
            type RF = <Scalar as Simd>::$reg;
            int_common!(UT, RF, $label);
            diff_unary!($label, UT, RF, neg, Tol::Exact);
            diff_unary!($label, UT, RF, abs, Tol::Exact);
            diff_varshift!($label, UT, RF, srav); // arithmetic (sign-extending) variable shift
        }
    };
    ($modname:ident, $ut_backend:ty, $reg:ident, $label:expr, unsigned) => {
        #[test]
        fn $modname() {
            type UT = <$ut_backend as Simd>::$reg;
            type RF = <Scalar as Simd>::$reg;
            int_common!(UT, RF, $label);
        }
    };
}

macro_rules! int_common {
    ($ut:ty, $rf:ty, $label:expr) => {{
        diff_binary!($label, $ut, $rf, add, Tol::Exact);
        diff_binary!($label, $ut, $rf, sub, Tol::Exact);
        diff_binary!($label, $ut, $rf, mul, Tol::Exact);
        diff_binary!($label, $ut, $rf, min, Tol::Exact);
        diff_binary!($label, $ut, $rf, max, Tol::Exact);
        diff_binary!($label, $ut, $rf, bitand, Tol::Exact);
        diff_binary!($label, $ut, $rf, bitor, Tol::Exact);
        diff_binary!($label, $ut, $rf, bitxor, Tol::Exact);
        diff_binary!($label, $ut, $rf, bitandnot, Tol::Exact);
        diff_unary!($label, $ut, $rf, not, Tol::Exact);
        diff_shift!($label, $ut, $rf, shl);
        diff_shift!($label, $ut, $rf, shr);
        diff_varshift!($label, $ut, $rf, shlv);
        diff_varshift!($label, $ut, $rf, shrv);
        diff_reduce!($label, $ut, $rf, sum_elements, Tol::Exact);
        diff_reduce!($label, $ut, $rf, prod_elements, Tol::Exact);
        diff_reduce!($label, $ut, $rf, min_element, Tol::Exact);
        diff_reduce!($label, $ut, $rf, max_element, Tol::Exact);
        diff_reduce!($label, $ut, $rf, wrapping_sum, Tol::Exact);
        diff_reduce!($label, $ut, $rf, wrapping_product, Tol::Exact);
    }};
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
use super::*;
use thermite::backend::x86_v1::X86V1;
use thermite::backend::x86_v2::X86V2;
use thermite::backend::x86_v3::X86V3;

// --- X86V3 (AVX2 + FMA) vs Scalar -----------------------------------------
mod v3_float {
    use super::*;
    float_reg_tests!(f32x4, X86V3, f32x4, "x86_v3 f32x4");
    float_reg_tests!(f32x8, X86V3, f32x8, "x86_v3 f32x8");
    float_reg_tests!(f32x16, X86V3, f32x16, "x86_v3 f32x16");
    float_reg_tests!(f64x2, X86V3, f64x2, "x86_v3 f64x2");
    float_reg_tests!(f64x4, X86V3, f64x4, "x86_v3 f64x4");
    float_reg_tests!(f64x8, X86V3, f64x8, "x86_v3 f64x8");
}
mod v3_int {
    use super::*;
    int_reg_tests!(i32x4, X86V3, i32x4, "x86_v3 i32x4", signed);
    int_reg_tests!(i32x8, X86V3, i32x8, "x86_v3 i32x8", signed);
    int_reg_tests!(i64x2, X86V3, i64x2, "x86_v3 i64x2", signed);
    int_reg_tests!(i64x4, X86V3, i64x4, "x86_v3 i64x4", signed);
    int_reg_tests!(u32x4, X86V3, u32x4, "x86_v3 u32x4", unsigned);
    int_reg_tests!(u32x8, X86V3, u32x8, "x86_v3 u32x8", unsigned);
    int_reg_tests!(u64x2, X86V3, u64x2, "x86_v3 u64x2", unsigned);
    int_reg_tests!(u64x4, X86V3, u64x4, "x86_v3 u64x4", unsigned);
}

// --- X86V2 (SSE4.2) vs Scalar ---------------------------------------------
mod v2_float {
    use super::*;
    float_reg_tests!(f32x4, X86V2, f32x4, "x86_v2 f32x4");
    float_reg_tests!(f32x8, X86V2, f32x8, "x86_v2 f32x8");
    float_reg_tests!(f64x2, X86V2, f64x2, "x86_v2 f64x2");
    float_reg_tests!(f64x4, X86V2, f64x4, "x86_v2 f64x4");
}
mod v2_int {
    use super::*;
    int_reg_tests!(i32x4, X86V2, i32x4, "x86_v2 i32x4", signed);
    int_reg_tests!(i32x8, X86V2, i32x8, "x86_v2 i32x8", signed);
    int_reg_tests!(i64x2, X86V2, i64x2, "x86_v2 i64x2", signed);
    int_reg_tests!(u32x4, X86V2, u32x4, "x86_v2 u32x4", unsigned);
    int_reg_tests!(u64x2, X86V2, u64x2, "x86_v2 u64x2", unsigned);
}

// --- X86V1 (SSE2) vs Scalar ------------------------------------------------
mod v1_float {
    use super::*;
    float_reg_tests!(f32x4, X86V1, f32x4, "x86_v1 f32x4");
    float_reg_tests!(f32x8, X86V1, f32x8, "x86_v1 f32x8");
    float_reg_tests!(f64x2, X86V1, f64x2, "x86_v1 f64x2");
    float_reg_tests!(f64x4, X86V1, f64x4, "x86_v1 f64x4");
}
mod v1_int {
    use super::*;
    int_reg_tests!(i32x4, X86V1, i32x4, "x86_v1 i32x4", signed);
    int_reg_tests!(i32x8, X86V1, i32x8, "x86_v1 i32x8", signed);
    int_reg_tests!(i64x2, X86V1, i64x2, "x86_v1 i64x2", signed);
    int_reg_tests!(u32x4, X86V1, u32x4, "x86_v1 u32x4", unsigned);
    int_reg_tests!(u64x2, X86V1, u64x2, "x86_v1 u64x2", unsigned);
}
}

// wasm: native f32x4/f64x2/i32x4/i64x2/u32x4/u64x2 (128-bit), wider via ArrayRegister.
#[cfg(target_arch = "wasm32")]
mod wasm {
use super::*;
use thermite::backend::wasm::Wasm;

mod wasm_float {
    use super::*;
    float_reg_tests!(f32x4, Wasm, f32x4, "wasm f32x4");
    float_reg_tests!(f32x8, Wasm, f32x8, "wasm f32x8");
    float_reg_tests!(f32x16, Wasm, f32x16, "wasm f32x16");
    float_reg_tests!(f64x2, Wasm, f64x2, "wasm f64x2");
    float_reg_tests!(f64x4, Wasm, f64x4, "wasm f64x4");
    float_reg_tests!(f64x8, Wasm, f64x8, "wasm f64x8");
}
mod wasm_int {
    use super::*;
    int_reg_tests!(i32x4, Wasm, i32x4, "wasm i32x4", signed);
    int_reg_tests!(i32x8, Wasm, i32x8, "wasm i32x8", signed);
    int_reg_tests!(i64x2, Wasm, i64x2, "wasm i64x2", signed);
    int_reg_tests!(i64x4, Wasm, i64x4, "wasm i64x4", signed);
    int_reg_tests!(u32x4, Wasm, u32x4, "wasm u32x4", unsigned);
    int_reg_tests!(u32x8, Wasm, u32x8, "wasm u32x8", unsigned);
    int_reg_tests!(u64x2, Wasm, u64x2, "wasm u64x2", unsigned);
    int_reg_tests!(u64x4, Wasm, u64x4, "wasm u64x4", unsigned);
}
}
