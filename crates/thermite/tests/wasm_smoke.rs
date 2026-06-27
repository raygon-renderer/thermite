//! Minimal smoke test that proves the wasm test pipeline end-to-end (compile a libtest binary
//! for `wasm32-wasip1`, run it under the `wasm-runner` wasmtime host, see pass/fail + exit code).
//!
//! Backend-generic: exercises the native SIMD backend for the target (`Wasm` on wasm, `X86V3`
//! on x86) so it both validates the wasm `u8x16`/`i16x8` registers and confirms the harness runs.
#![cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32"))]

use thermite::register::{IntegerRegister, NumericRegister, Register};
use thermite::simd::{NativeSimd, Simd};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
type Backend = thermite::backend::x86_v3::X86V3;
#[cfg(target_arch = "wasm32")]
type Backend = thermite::backend::wasm::Wasm;

#[test]
fn u8x16_add() {
    type R = <Backend as Simd>::u8x16;
    let c = R::add(R::splat(200), R::splat(7));
    assert_eq!(R::as_array(&c).as_slice(), &[207u8; 16]);
}

#[test]
fn u8x16_wrapping_add() {
    type R = <Backend as Simd>::u8x16;
    // 200 + 100 wraps to 44 in u8.
    let c = R::add(R::splat(200), R::splat(100));
    assert_eq!(R::as_array(&c).as_slice(), &[44u8; 16]);
}

#[test]
fn u8x16_popcnt() {
    type R = <Backend as Simd>::u8x16;
    let c = R::count_ones(R::splat(0b1011_0001));
    assert_eq!(R::as_array(&c).as_slice(), &[4u8; 16]);
}

#[test]
fn u8x16_reverse() {
    type R = <Backend as Simd>::u8x16;
    let v = R::indexed(); // [0, 1, ..., 15]
    let r = R::reverse(v);
    let want: [u8; 16] = core::array::from_fn(|i| (15 - i) as u8);
    assert_eq!(R::as_array(&r).as_slice(), &want);
}

#[test]
fn i16x8_mul() {
    type R = <Backend as Simd>::i16x8;
    let c = R::mul(R::splat(300), R::splat(3));
    assert_eq!(R::as_array(&c).as_slice(), &[900i16; 8]);
}
