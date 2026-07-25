//! Interleave / deinterleave correctness for the native 8-bit registers (v1 + v2 + v3).
//!
//! Checks (1) the interleave lane layout, (2) that deinterleave inverts interleave, and
//! (3) deinterleave directly against a scalar even/odd oracle. This guards the v2 `pshufb`
//! deinterleave and the v3 AVX2 cross-lane `unpack`/`permute4x64` sequences (and the v1
//! scalar fallback).
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use generic_array::typenum::Unsigned;
use thermite::register::{CoreRegister, InterleaveRegister as _};
// Which of these are used varies by backend cfg (x86 / wasm / neon).
#[allow(unused_imports)]
use thermite::simd::{NativeSimd, Simd};

macro_rules! interleave_roundtrip {
    ($name:ident, $reg:ty, $e:ty) => {
        #[test]
        fn $name() {
            type UT = $reg;
            let lanes = <<UT as CoreRegister>::Lanes as Unsigned>::USIZE;
            let half = lanes / 2;

            // Distinct values for a/b that stay within the byte range (wrapping `as` is fine -
            // both sides use the same cast, so comparisons stay consistent).
            let a: Vec<$e> = (0..lanes).map(|i| i as $e).collect();
            let b: Vec<$e> = (0..lanes).map(|i| (i as u32 ^ 0xA5) as $e).collect();
            let av = harness::make_array::<UT>(&a);
            let bv = harness::make_array::<UT>(&b);

            // (1) interleave layout: lo = [a0,b0,a1,b1,...], hi = upper half likewise.
            let (lo, hi) = <UT>::interleave(av, bv);
            let lo_r = harness::read::<UT>(&lo);
            let hi_r = harness::read::<UT>(&hi);
            for k in 0..half {
                assert_eq!(lo_r[2 * k], a[k], "interleave lo even lane {k}");
                assert_eq!(lo_r[2 * k + 1], b[k], "interleave lo odd lane {k}");
                assert_eq!(hi_r[2 * k], a[k + half], "interleave hi even lane {k}");
                assert_eq!(hi_r[2 * k + 1], b[k + half], "interleave hi odd lane {k}");
            }

            // (2) deinterleave inverts interleave.
            let (da, db) = <UT>::deinterleave(lo, hi);
            assert_eq!(harness::read::<UT>(&da), a, "deinterleave(interleave) a");
            assert_eq!(harness::read::<UT>(&db), b, "deinterleave(interleave) b");

            // (3) deinterleave(x, y) == (even-indexed of [x++y], odd-indexed of [x++y]).
            let (de, dodd) = <UT>::deinterleave(av, bv);
            let cat: Vec<$e> = a.iter().chain(b.iter()).copied().collect();
            let evens: Vec<$e> = (0..lanes).map(|i| cat[2 * i]).collect();
            let odds: Vec<$e> = (0..lanes).map(|i| cat[2 * i + 1]).collect();
            assert_eq!(harness::read::<UT>(&de), evens, "deinterleave evens");
            assert_eq!(harness::read::<UT>(&dodd), odds, "deinterleave odds");
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    use thermite::backend::x86_v1::X86V1;
    use thermite::backend::x86_v2::X86V2;
    use thermite::backend::x86_v3::X86V3;

    interleave_roundtrip!(v1_i8x16, <X86V1 as NativeSimd>::i8xN, i8);
    interleave_roundtrip!(v1_u8x16, <X86V1 as NativeSimd>::u8xN, u8);
    interleave_roundtrip!(v2_i8x16, <X86V2 as NativeSimd>::i8xN, i8);
    interleave_roundtrip!(v2_u8x16, <X86V2 as NativeSimd>::u8xN, u8);
    interleave_roundtrip!(v3_i8x32, <X86V3 as NativeSimd>::i8xN, i8);
    interleave_roundtrip!(v3_u8x32, <X86V3 as NativeSimd>::u8xN, u8);
    interleave_roundtrip!(v3_i8x16, <X86V3 as Simd>::i8x16, i8);
    interleave_roundtrip!(v3_u8x16, <X86V3 as Simd>::u8x16, u8);
}

// WASM: native 16-lane i8x16/u8x16 (= the fixed i8x16 slot).
#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use thermite::backend::wasm::Wasm;

    interleave_roundtrip!(i8x16, <Wasm as NativeSimd>::i8xN, i8);
    interleave_roundtrip!(u8x16, <Wasm as NativeSimd>::u8xN, u8);
}

// NEON: native 16-lane i8x16/u8x16 (= the fixed i8x16 slot).
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use thermite::backend::neon::Neon;

    interleave_roundtrip!(i8x16, <Neon as NativeSimd>::i8xN, i8);
    interleave_roundtrip!(u8x16, <Neon as NativeSimd>::u8xN, u8);
}
