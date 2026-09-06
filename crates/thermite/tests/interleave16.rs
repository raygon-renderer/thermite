//! Interleave / deinterleave correctness for the 16-bit registers on every backend.
//!
//! Checks (1) the interleave lane layout, (2) that deinterleave inverts interleave, and
//! (3) deinterleave directly against a scalar even/odd oracle. This is what guards the
//! AVX2 `pshufb`/`unpack`/`permute` deinterleave sequences and the AVX-512 `vpermw`
//! forms.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use generic_array::typenum::Unsigned;
use thermite::register::{CoreRegister, InterleaveRegister as _};
use thermite::simd::{NativeSimd, Simd};

macro_rules! interleave_roundtrip {
    ($reg:ty, $e:ty) => {{
        let lanes = <<$reg as CoreRegister>::Lanes as Unsigned>::USIZE;
        let half = lanes / 2;

        let a: Vec<$e> = (0..lanes).map(|i| i as $e).collect();
        let b: Vec<$e> = (0..lanes).map(|i| (i + 100) as $e).collect();
        let av = harness::make_array::<$reg>(&a);
        let bv = harness::make_array::<$reg>(&b);

        // (1) interleave layout: lo = [a0,b0,a1,b1,...], hi = upper half likewise.
        let (lo, hi) = <$reg>::interleave(av, bv);
        let lo_r = harness::read::<$reg>(&lo);
        let hi_r = harness::read::<$reg>(&hi);
        for k in 0..half {
            assert_eq!(lo_r[2 * k], a[k], "interleave lo even lane {k}");
            assert_eq!(lo_r[2 * k + 1], b[k], "interleave lo odd lane {k}");
            assert_eq!(hi_r[2 * k], a[k + half], "interleave hi even lane {k}");
            assert_eq!(hi_r[2 * k + 1], b[k + half], "interleave hi odd lane {k}");
        }

        // (2) deinterleave inverts interleave.
        let (da, db) = <$reg>::deinterleave(lo, hi);
        assert_eq!(harness::read::<$reg>(&da), a, "deinterleave(interleave) a");
        assert_eq!(harness::read::<$reg>(&db), b, "deinterleave(interleave) b");

        // (3) deinterleave(x, y) == (even-indexed of [x++y], odd-indexed of [x++y]).
        let (de, dodd) = <$reg>::deinterleave(av, bv);
        let cat: Vec<$e> = a.iter().chain(b.iter()).copied().collect();
        let evens: Vec<$e> = (0..lanes).map(|i| cat[2 * i]).collect();
        let odds: Vec<$e> = (0..lanes).map(|i| cat[2 * i + 1]).collect();
        assert_eq!(harness::read::<$reg>(&de), evens, "deinterleave evens");
        assert_eq!(harness::read::<$reg>(&dodd), odds, "deinterleave odds");
    }};
}

for_each_backend! {
    fn x8<S: Simd>() {
        interleave_roundtrip!(<S as Simd>::i16x8, i16);
        interleave_roundtrip!(<S as Simd>::u16x8, u16);
    }
    fn x16<S: Simd>() {
        interleave_roundtrip!(<S as Simd>::i16x16, i16);
        interleave_roundtrip!(<S as Simd>::u16x16, u16);
    }
    /// The native-width word registers (8 lanes on 128-bit backends, 16 on AVX2, 32 on AVX-512).
    fn native<S: Simd>() {
        interleave_roundtrip!(<S as NativeSimd>::i16xN, i16);
        interleave_roundtrip!(<S as NativeSimd>::u16xN, u16);
    }
    fn reduced<S: Simd>() {
        interleave_roundtrip!(<S as Simd>::i16x4, i16);
        interleave_roundtrip!(<S as Simd>::u16x4, u16);
    }
}
