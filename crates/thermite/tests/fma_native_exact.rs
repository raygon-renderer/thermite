//! Backend-generic FMA exactness spot checks with precomputed expected bits, so
//! they run on targets with no hardware oracle, most importantly wasm under
//! wasmtime, where `mul_add` dispatches between the engine's relaxed madd (when
//! a one-time canary proves it fused) and the round-to-odd emulation. Every
//! case here has a hand-derivable single-rounding answer. A canary that wrongly
//! claims "fused" on a multiply-then-add engine fails the midpoint cases.
//!
//! (`fma_exact.rs` is the deep suite, bit-compared against the hardware FMA
//! instruction on x86.)
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use thermite::register::{FloatRegister, Register};
use thermite::simd::Simd;

// Every backend: hardware FMA where it exists, the correctly rounded emulation
// elsewhere. Both must be bit-exact, so the same assertions apply to all rows.
#[inline(always)]
fn fma64<S: Simd>(a: f64, b: f64, c: f64) -> f64 {
    <S::f64x2>::as_slice(&<S::f64x2>::mul_add(
        <S::f64x2>::splat(a),
        <S::f64x2>::splat(b),
        <S::f64x2>::splat(c),
    ))[0]
}

#[inline(always)]
fn fma32<S: Simd>(a: f32, b: f32, c: f32) -> f32 {
    <S::f32x4>::as_slice(&<S::f32x4>::mul_add(
        <S::f32x4>::splat(a),
        <S::f32x4>::splat(b),
        <S::f32x4>::splat(c),
    ))[0]
}

macro_rules! ctx {
    () => {
        // `#[inline(always)]`: these live inside a dispatched body, and a plain
        // fn here compiled featureless (thermite-audit, 2026-09-06).
        #[allow(dead_code)]
        #[inline(always)]
        fn fma64(a: f64, b: f64, c: f64) -> f64 {
            super::super::fma64::<S>(a, b, c)
        }
        #[allow(dead_code)]
        #[inline(always)]
        fn fma32(a: f32, b: f32, c: f32) -> f32 {
            super::super::fma32::<S>(a, b, c)
        }
    };
}

for_each_backend_concrete! {

/// The case that separates a fused multiply-add from multiply-then-add:
/// (1 + 2^-27)(1 - 2^-27) = 1 - 2^-54 exactly, the midpoint between
/// 1 - 2^-53 and 1. Fused: -2^-54. Unfused: 0.
fn f64_midpoint() {
    ctx!();
    let a = 1.0 + 2.0_f64.powi(-27);
    let b = 1.0 - 2.0_f64.powi(-27);
    assert_eq!(fma64(a, b, -1.0).to_bits(), (-(2.0_f64.powi(-54))).to_bits());
    assert_eq!(fma64(a, b, 1.0).to_bits(), 2.0_f64.to_bits()); // 2 - 2^-54 rounds to 2
}

fn f32_midpoint() {
    ctx!();
    let a = 1.0 + 2.0_f32.powi(-12);
    let b = 1.0 - 2.0_f32.powi(-12);
    assert_eq!(fma32(a, b, -1.0).to_bits(), (-(2.0_f32.powi(-24))).to_bits());
}

/// Exact-zero signs per IEEE 754: -0 only when product and addend are both
/// negative zeros. Exact cancellation of nonzero values gives +0.
fn signed_zeros() {
    ctx!();
    assert_eq!(fma64(-0.0, 3.0, -0.0).to_bits(), (-0.0_f64).to_bits());
    assert_eq!(fma64(0.0, 3.0, -0.0).to_bits(), 0.0_f64.to_bits());
    assert_eq!(fma64(2.0, 3.0, -6.0).to_bits(), 0.0_f64.to_bits());
    assert_eq!(fma32(-0.0, 3.0, -0.0).to_bits(), (-0.0_f32).to_bits());
}

/// Subnormal addend against an ordinary product must neither perturb nor trap.
fn subnormal_c() {
    ctx!();
    assert_eq!(fma64(1.5, 3.0, f64::from_bits(1)).to_bits(), 4.5_f64.to_bits());
}

/// Subnormal-range results, computed exactly (no flush-to-zero on this path).
#[cfg(not(feature = "ignore_denormals"))]
fn subnormal_results() {
    ctx!();
    // 2^-537 * 2^-538 = 2^-1075, exactly half the smallest subnormal: ties to 0.
    assert_eq!(
        fma64(2.0_f64.powi(-537), 2.0_f64.powi(-538), 0.0).to_bits(),
        0.0_f64.to_bits()
    );
    // MIN_POSITIVE * (1 - 2^-53) sits exactly on the top-subnormal/min-normal
    // midpoint: ties up (min-normal's mantissa is even).
    let b = 1.0 - 2.0_f64.powi(-53);
    assert_eq!(fma64(f64::MIN_POSITIVE, b, 0.0).to_bits(), f64::MIN_POSITIVE.to_bits());
    // f32: 2^-75 * 2^-74 = 2^-149, the smallest f32 subnormal, exactly.
    assert_eq!(
        fma32(2.0_f32.powi(-75), 2.0_f32.powi(-74), 0.0).to_bits(),
        2.0_f32.powi(-149).to_bits()
    );
}

/// Specials route correctly whatever the lowering.
fn specials() {
    ctx!();
    assert_eq!(fma64(f64::MAX, 2.0, -f64::MAX).to_bits(), f64::MAX.to_bits());
    assert!(fma64(f64::INFINITY, 0.0, 1.0).is_nan());
    assert_eq!(fma64(f64::MAX, 2.0, f64::NEG_INFINITY), f64::NEG_INFINITY);
    assert!(fma64(f64::NAN, 1.0, 2.0).is_nan());
}

}
