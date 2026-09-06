//! `GenericMask::from_native_bitmask` / `from_bitmask` correctness.
//!
//! The inverses of `native_bitmask` / `bitmask`: a packed bit per lane turned
//! back into a lane mask. Checked three ways over the same battery of bit
//! patterns `mask_scan` uses (empty, full, every single bit, full-minus-one-bit):
//!
//!   - against an independently built mask (elements set from the pattern, then
//!     `cmp_ne(ZERO)`), so a wrong lane order or a dropped lane shows up,
//!   - round-tripped through `native_bitmask` / `bitmask`,
//!   - with bits *above* the lane count set, which every implementation must
//!     ignore (`ArrayRegister` relies on it when it shifts one word per
//!     sub-register), and with a bit slice *shorter* than the lane count, whose
//!     unreached lanes must come back `false`.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use thermite::mask::GenericMask;
use thermite::prelude::*;

macro_rules! check {
    ($vty:ty) => {{
        type V = $vty;
        type M = <$vty as GenericVector>::Mask;
        const LANES: usize = <V as GenericVector>::LANES;

        let make = |bits: u64| -> M {
            let mut data = [<V as GenericVector>::Element::default(); LANES];
            for lane in 0..LANES {
                if (bits >> lane) & 1 == 1 {
                    data[lane] = 1 as _;
                }
            }
            V::new(data.into()).cmp_ne(V::ZERO)
        };

        let same = |a: M, b: M| -> bool { a.select(V::ONE, V::ZERO).cmp_eq(b.select(V::ONE, V::ZERO)).all() };

        let full: u64 = if LANES >= 64 { u64::MAX } else { (1u64 << LANES) - 1 };

        let mut patterns = vec![0u64, full];
        for k in 0..LANES {
            patterns.push(1u64 << k);
            patterns.push(full & !(1u64 << k));
        }

        for &bits in &patterns {
            let want = make(bits);

            let got = M::from_native_bitmask(bits);
            assert!(same(got, want), "from_native_bitmask({bits:#x}) lanes={LANES}");

            // bits above the lane count must be ignored
            let noisy = M::from_native_bitmask(bits | !full);
            assert!(
                same(noisy, want),
                "from_native_bitmask({:#x}) lanes={LANES}",
                bits | !full
            );

            if let Some(bm) = want.native_bitmask() {
                assert_eq!(
                    bm & full,
                    bits,
                    "native_bitmask round-trip bits={bits:#x} lanes={LANES}"
                );
                assert!(
                    same(M::from_native_bitmask(bm), want),
                    "from_native_bitmask(native_bitmask) bits={bits:#x} lanes={LANES}"
                );
            }

            #[cfg(feature = "bitvec")]
            {
                let packed = want.bitmask();
                assert!(
                    same(M::from_bitmask(&packed), want),
                    "from_bitmask(bitmask) bits={bits:#x} lanes={LANES}"
                );

                // a short slice leaves the unreached lanes false
                let half = LANES / 2;
                let truncated = M::from_bitmask(&packed[..half]);
                assert!(
                    same(truncated, make(bits & ((1u64 << half) - 1))),
                    "from_bitmask(bitmask[..{half}]) bits={bits:#x} lanes={LANES}"
                );
            }
        }
    }};
}

for_each_backend_concrete! {
    fn f32x4() { check!(thermite::simd::f32x4<S>) }
    fn f32x8() { check!(thermite::simd::f32x8<S>) }
    fn f32x16() { check!(thermite::simd::f32x16<S>) }
    fn f64x2() { check!(thermite::simd::f64x2<S>) }
    fn f64x4() { check!(thermite::simd::f64x4<S>) }
    fn f64x8() { check!(thermite::simd::f64x8<S>) }
    fn u8x8() { check!(thermite::simd::u8x8<S>) }
    fn u8x16() { check!(thermite::simd::u8x16<S>) }
    fn i8x16() { check!(thermite::simd::i8x16<S>) }
    fn u16x8() { check!(thermite::simd::u16x8<S>) }
    fn u16x16() { check!(thermite::simd::u16x16<S>) }
    fn u32x3() { check!(thermite::simd::u32x3<S>) } // reduced register
    fn u32x4() { check!(thermite::simd::u32x4<S>) }
    fn u32x8() { check!(thermite::simd::u32x8<S>) }
    fn u32x16() { check!(thermite::simd::u32x16<S>) }
    fn u64x2() { check!(thermite::simd::u64x2<S>) }
    fn u64x4() { check!(thermite::simd::u64x4<S>) }
    fn u64x8() { check!(thermite::simd::u64x8<S>) }
}
