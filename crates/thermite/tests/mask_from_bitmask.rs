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
//!
//! Covers the scalar backend (native 1-lane, `ArrayRegister`, and reduced
//! register masks), the x86 v1/v2/v3 paths, and NEON on aarch64.

use thermite::backend::scalar::Scalar;
use thermite::mask::GenericMask;
use thermite::prelude::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use thermite::backend::{x86_v1::X86V1, x86_v2::X86V2, x86_v3::X86V3};

macro_rules! check {
    ($name:ident, $vty:ty) => {
        #[test]
        fn $name() {
            type V = $vty;
            type M = <$vty as GenericVector>::Mask;
            const LANES: usize = <V as GenericVector>::LANES;

            // Independent mask construction: lane set -> element 1, else 0,
            // then `!= 0` lifts it into the mask domain.
            let make = |bits: u64| -> M {
                let mut data = [<V as GenericVector>::Element::default(); LANES];
                for lane in 0..LANES {
                    if (bits >> lane) & 1 == 1 {
                        data[lane] = 1 as _;
                    }
                }
                V::new(data.into()).cmp_ne(V::ZERO)
            };

            // Masks are compared as the vectors they select, which needs no
            // per-lane mask accessor and fails loudly on a lane permutation.
            let same = |a: M, b: M| -> bool { a.select(V::ONE, V::ZERO).cmp_eq(b.select(V::ONE, V::ZERO)).all() };

            let full: u64 = if LANES >= 64 {
                u64::MAX
            } else {
                (1u64 << LANES) - 1
            };

            let mut patterns = vec![0u64, full];
            for k in 0..LANES {
                patterns.push(1u64 << k);
                patterns.push(full & !(1u64 << k));
            }

            for &bits in &patterns {
                let want = make(bits);

                let got = M::from_native_bitmask(bits);
                assert!(same(got, want), "from_native_bitmask({bits:#x}) lanes={LANES}");

                // Bits above the lane count must not leak into a lane.
                let noisy = M::from_native_bitmask(bits | !full);
                assert!(
                    same(noisy, want),
                    "from_native_bitmask({:#x}) lanes={LANES}",
                    bits | !full
                );

                // native_bitmask round-trip, where the backend has one.
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

                // bitvec round-trip.
                #[cfg(feature = "bitvec")]
                {
                    let packed = want.bitmask();
                    assert!(
                        same(M::from_bitmask(&packed), want),
                        "from_bitmask(bitmask) bits={bits:#x} lanes={LANES}"
                    );

                    // A short slice leaves the lanes it does not reach `false`.
                    let half = LANES / 2;
                    let truncated = M::from_bitmask(&packed[..half]);
                    assert!(
                        same(truncated, make(bits & ((1u64 << half) - 1))),
                        "from_bitmask(bitmask[..{half}]) bits={bits:#x} lanes={LANES}"
                    );
                }
            }
        }
    };
}

macro_rules! suite {
    ($modname:ident, $backend:ty) => {
        mod $modname {
            use super::*;
            check!(f32x4, thermite::simd::f32x4<$backend>);
            check!(f32x8, thermite::simd::f32x8<$backend>);
            check!(f32x16, thermite::simd::f32x16<$backend>);
            check!(f64x2, thermite::simd::f64x2<$backend>);
            check!(f64x4, thermite::simd::f64x4<$backend>);
            check!(u8x8, thermite::simd::u8x8<$backend>);
            check!(u8x16, thermite::simd::u8x16<$backend>);
            check!(i8x16, thermite::simd::i8x16<$backend>);
            check!(u16x8, thermite::simd::u16x8<$backend>);
            check!(u16x16, thermite::simd::u16x16<$backend>);
            check!(u32x3, thermite::simd::u32x3<$backend>); // reduced register
            check!(u32x4, thermite::simd::u32x4<$backend>);
            check!(u32x8, thermite::simd::u32x8<$backend>);
            check!(u32x16, thermite::simd::u32x16<$backend>);
            check!(u64x2, thermite::simd::u64x2<$backend>);
            check!(u64x4, thermite::simd::u64x4<$backend>);
        }
    };
}

suite!(scalar, Scalar);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
suite!(v1, X86V1);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
suite!(v2, X86V2);
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
suite!(v3, X86V3);

#[cfg(target_arch = "wasm32")]
suite!(wasm, thermite::backend::wasm::Wasm);

#[cfg(target_arch = "aarch64")]
suite!(neon, thermite::backend::neon::Neon);
