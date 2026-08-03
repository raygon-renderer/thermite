//! Interleave / deinterleave correctness for the native 16-bit registers (v2 + v3).
//!
//! Checks (1) the interleave lane layout, (2) that deinterleave inverts interleave, and
//! (3) deinterleave directly against a scalar even/odd oracle. This is what guards the
//! AVX2 `pshufb`/`unpack`/`permute` deinterleave sequences.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use generic_array::typenum::Unsigned;
use thermite::register::{CoreRegister, InterleaveRegister as _};
use thermite::simd::{Simd};

macro_rules! interleave_roundtrip {
    ($name:ident, $backend:ty, $reg:ident, $e:ty) => {
        #[test]
        fn $name() {
            type UT = <$backend as Simd>::$reg;
            let lanes = <<UT as CoreRegister>::Lanes as Unsigned>::USIZE;
            let half = lanes / 2;

            let a: Vec<$e> = (0..lanes).map(|i| i as $e).collect();
            let b: Vec<$e> = (0..lanes).map(|i| (i + 100) as $e).collect();
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
    use thermite::backend::x86_v2::X86V2;
    use thermite::backend::x86_v3::X86V3;

    interleave_roundtrip!(v2_i16x8, X86V2, i16x8, i16);
    interleave_roundtrip!(v2_u16x8, X86V2, u16x8, u16);
    interleave_roundtrip!(v3_i16x8, X86V3, i16x8, i16);
    interleave_roundtrip!(v3_u16x8, X86V3, u16x8, u16);
    interleave_roundtrip!(v3_i16x16, X86V3, i16x16, i16);
    interleave_roundtrip!(v3_u16x16, X86V3, u16x16, u16);
}

// WASM: native 8-lane i16x8 (= i16xN) + the emulated 16-lane i16x16 (ArrayRegister).
#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use thermite::backend::wasm::Wasm;

    interleave_roundtrip!(i16x8, Wasm, i16x8, i16);
    interleave_roundtrip!(u16x8, Wasm, u16x8, u16);
    interleave_roundtrip!(i16x16, Wasm, i16x16, i16);
    interleave_roundtrip!(u16x16, Wasm, u16x16, u16);
}

// NEON: native 8-lane i16x8 (= i16xN) + the emulated 16-lane i16x16 (ArrayRegister).
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use thermite::backend::neon::Neon;

    interleave_roundtrip!(i16x8, Neon, i16x8, i16);
    interleave_roundtrip!(u16x8, Neon, u16x8, u16);
    interleave_roundtrip!(i16x16, Neon, i16x16, i16);
    interleave_roundtrip!(u16x16, Neon, u16x16, u16);
}
