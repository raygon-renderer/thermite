//! `GenericMask::first_set` / `last_set` / `count_set` correctness.
//!
//! These turn a mask into scanning primitives (find-first / find-last / popcount
//! of true lanes). Verified against a per-lane oracle over a battery of bit
//! patterns - empty, full, every single bit, and full-minus-one-bit - which
//! pins down the first/last/count edges. Exercised across the scalar backend
//! (native 1-lane, `ArrayRegister`, and reduced register masks) and, on x86,
//! the native movemask paths of v1/v2/v3.

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
            const LANES: usize = <V as GenericVector>::LANES;

            // Build the mask for a bit pattern: lane set -> element 1, else 0,
            // then `!= 0` lifts it into the mask domain.
            let make = |bits: u64| -> <V as GenericVector>::Mask {
                let mut data = [<V as GenericVector>::Element::default(); LANES];
                for lane in 0..LANES {
                    if (bits >> lane) & 1 == 1 {
                        data[lane] = 1 as _;
                    }
                }
                V::new(data.into()).cmp_ne(V::ZERO)
            };

            let full: u64 = if LANES >= 64 { u64::MAX } else { (1u64 << LANES) - 1 };

            // Patterns: empty, full, each single bit, and full with one bit cleared.
            let mut patterns = alloc_patterns(full, LANES);
            for bits in patterns.drain(..) {
                let m = make(bits);

                // Oracle over the LANES low bits.
                let mut first = None;
                let mut last = None;
                let mut count = 0usize;
                for lane in 0..LANES {
                    if (bits >> lane) & 1 == 1 {
                        if first.is_none() {
                            first = Some(lane);
                        }
                        last = Some(lane);
                        count += 1;
                    }
                }

                assert_eq!(m.first_set(), first, "first_set bits={bits:#x} lanes={LANES}");
                assert_eq!(m.last_set(), last, "last_set bits={bits:#x} lanes={LANES}");
                assert_eq!(m.count_set(), count, "count_set bits={bits:#x} lanes={LANES}");
            }
        }
    };
}

// Collect the test patterns into a Vec so the closure body stays simple.
fn alloc_patterns(full: u64, lanes: usize) -> Vec<u64> {
    let mut v = vec![0u64, full];
    for k in 0..lanes {
        v.push(1u64 << k); // single bit
        v.push(full & !(1u64 << k)); // full minus one bit
    }
    v
}

macro_rules! suite {
    ($modname:ident, $backend:ty) => {
        mod $modname {
            use super::*;
            check!(u8x8, thermite::simd::u8x8<$backend>);
            check!(u8x16, thermite::simd::u8x16<$backend>);
            check!(u16x8, thermite::simd::u16x8<$backend>);
            check!(u16x16, thermite::simd::u16x16<$backend>);
            check!(u32x3, thermite::simd::u32x3<$backend>); // reduced register
            check!(u32x4, thermite::simd::u32x4<$backend>);
            check!(u32x8, thermite::simd::u32x8<$backend>);
            check!(u32x16, thermite::simd::u32x16<$backend>);
            check!(u64x2, thermite::simd::u64x2<$backend>);
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
