//! Differential tests: every SIMD backend register op vs. the `Scalar`
//! reference, across the element-type x width matrix.
//!
//! See `harness/mod.rs` for the methodology. Each suite below is written once
//! over `S: Simd`. `for_each_backend!` stamps it per compiled backend behind a
//! runtime ISA gate, so every backend (including the emulated wide slots on the
//! 128-bit ones) runs every row.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

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

// Per-lane variable shift coverage now comes from the shared `diff_varshift!`
// stamper in the harness, so the 8-bit suites can use it too.

// ---------------------------------------------------------------------------
// Float register suite, one slot of backend `$S` vs the same slot of Scalar.
// ---------------------------------------------------------------------------
macro_rules! float_ops {
    ($S:ty, $reg:ident) => {{
        let label = harness::label::<$S>(stringify!($reg));
        let label = label.as_str();

        // IEEE-correctly-rounded ops: must be bit-exact vs. scalar.
        diff_binary!(label, <$S as Simd>::$reg, <Scalar as Simd>::$reg, add, Tol::Exact);
        diff_binary!(label, <$S as Simd>::$reg, <Scalar as Simd>::$reg, sub, Tol::Exact);
        diff_binary!(label, <$S as Simd>::$reg, <Scalar as Simd>::$reg, mul, Tol::Exact);
        diff_binary!(label, <$S as Simd>::$reg, <Scalar as Simd>::$reg, div, Tol::Exact);
        diff_unary!(label, <$S as Simd>::$reg, <Scalar as Simd>::$reg, sqrt, Tol::Exact);

        // Sign / ordering: thermite-defined, scalar backend is the oracle.
        diff_unary!(label, <$S as Simd>::$reg, <Scalar as Simd>::$reg, neg, Tol::Exact);
        diff_unary!(label, <$S as Simd>::$reg, <Scalar as Simd>::$reg, abs, Tol::Exact);
        // Rel(0.0) == exact, except it treats +0.0 and -0.0 as equal
        // (numerically they are, and which signed zero min/max returns is
        // unspecified and differs harmlessly between backends).
        diff_binary_finite!(label, <$S as Simd>::$reg, <Scalar as Simd>::$reg, min, Tol::Rel(0.0));
        diff_binary_finite!(label, <$S as Simd>::$reg, <Scalar as Simd>::$reg, max, Tol::Rel(0.0));
        diff_unary!(label, <$S as Simd>::$reg, <Scalar as Simd>::$reg, floor, Tol::Exact);
        diff_unary!(label, <$S as Simd>::$reg, <Scalar as Simd>::$reg, ceil, Tol::Exact);
        diff_unary!(label, <$S as Simd>::$reg, <Scalar as Simd>::$reg, trunc, Tol::Exact);
        // NOTE: `round` is intentionally excluded, as its half-way rounding
        // direction diverges between backends (scalar = half-away-from-zero,
        // x86 = half-to-even).

        // NOTE: `rcp`/`rsqrt` are hardware approximations and flush
        // denormals, so they are accuracy-tested as a property (rcp(x)*x ≈ 1)
        // in `approx_recip.rs`, not differentially against exact scalar.

        // Horizontal reductions. `sum_elements` is non-associative so its
        // tree-vs-fold rounding diverges on adversarial inputs, so it gets a
        // tame-input accuracy test in `approx_recip.rs` instead. min/max
        // are associative, so they must agree (modulo NaN).
        diff_reduce_finite!(
            label,
            <$S as Simd>::$reg,
            <Scalar as Simd>::$reg,
            min_element,
            Tol::Rel(0.0)
        );
        diff_reduce_finite!(
            label,
            <$S as Simd>::$reg,
            <Scalar as Simd>::$reg,
            max_element,
            Tol::Rel(0.0)
        );
    }};
}

// ---------------------------------------------------------------------------
// Integer register suite. Every op below is correct on every integer
// width/backend. The two defects the harness originally found here were
// 32-bit reductions and 64-bit `mul`.
// ---------------------------------------------------------------------------
macro_rules! int_ops {
    ($S:ty, $reg:ident, signed) => {{
        let label = harness::label::<$S>(stringify!($reg));
        let label = label.as_str();
        int_common!(label, <$S as Simd>::$reg, <Scalar as Simd>::$reg);
        diff_unary!(label, <$S as Simd>::$reg, <Scalar as Simd>::$reg, neg, Tol::Exact);
        diff_unary!(label, <$S as Simd>::$reg, <Scalar as Simd>::$reg, abs, Tol::Exact);
        // arithmetic (sign-extending) variable shift
        diff_varshift!(label, <$S as Simd>::$reg, <Scalar as Simd>::$reg, srav);
    }};
    ($S:ty, $reg:ident, unsigned) => {{
        let label = harness::label::<$S>(stringify!($reg));
        let label = label.as_str();
        int_common!(label, <$S as Simd>::$reg, <Scalar as Simd>::$reg);
    }};
}

macro_rules! int_common {
    ($label:expr, $ut:ty, $rf:ty) => {{
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

// ---------------------------------------------------------------------------
// One #[test] per (backend, slot). Native on some backends, ArrayRegister-
// emulated on the rest. Both must agree with Scalar.
// ---------------------------------------------------------------------------
for_each_backend! {
    fn float_f32x4<S: Simd>() { float_ops!(S, f32x4) }
    fn float_f32x8<S: Simd>() { float_ops!(S, f32x8) }
    fn float_f32x16<S: Simd>() { float_ops!(S, f32x16) }
    fn float_f64x2<S: Simd>() { float_ops!(S, f64x2) }
    fn float_f64x4<S: Simd>() { float_ops!(S, f64x4) }
    fn float_f64x8<S: Simd>() { float_ops!(S, f64x8) }

    fn int_i32x4<S: Simd>() { int_ops!(S, i32x4, signed) }
    fn int_i32x8<S: Simd>() { int_ops!(S, i32x8, signed) }
    fn int_i32x16<S: Simd>() { int_ops!(S, i32x16, signed) }
    fn int_i64x2<S: Simd>() { int_ops!(S, i64x2, signed) }
    fn int_i64x4<S: Simd>() { int_ops!(S, i64x4, signed) }
    fn int_i64x8<S: Simd>() { int_ops!(S, i64x8, signed) }
    fn int_u32x4<S: Simd>() { int_ops!(S, u32x4, unsigned) }
    fn int_u32x8<S: Simd>() { int_ops!(S, u32x8, unsigned) }
    fn int_u32x16<S: Simd>() { int_ops!(S, u32x16, unsigned) }
    fn int_u64x2<S: Simd>() { int_ops!(S, u64x2, unsigned) }
    fn int_u64x4<S: Simd>() { int_ops!(S, u64x4, unsigned) }
    fn int_u64x8<S: Simd>() { int_ops!(S, u64x8, unsigned) }
}
