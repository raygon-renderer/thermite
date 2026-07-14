//! `UnsignedIntegerVector::abs_diff` correctness.
//!
//! `a.abs_diff(b)` is the per-lane unsigned `|a - b|`, computed branchlessly as
//! `(a -| b) | (b -| a)` with saturating subtraction. Verified against Rust's
//! `uN::abs_diff` oracle over all ordered pairs of a probe set (covering equal,
//! a<b, a>b, and the 0/MAX corners). The default composes `saturating_sub`,
//! which is natively overridden per backend, so this is exercised across scalar
//! and x86 v1/v2/v3.

use thermite::backend::scalar::Scalar;
use thermite::prelude::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use thermite::backend::{x86_v1::X86V1, x86_v2::X86V2, x86_v3::X86V3};

macro_rules! check {
    ($name:ident, $vty:ty, $elem:ty, $n:literal) => {
        #[test]
        fn $name() {
            type V = $vty;
            const N: usize = $n;

            let max = <$elem>::MAX;
            let probes: [$elem; 8] = [0, 1, 2, 127, 128, max / 2, max - 1, max];
            let p = probes.len();

            // Every ordered (probe, rotated-probe) pair across the lanes.
            for shift in 0..p {
                let mut da = [0 as $elem; N];
                let mut db = [0 as $elem; N];
                for i in 0..N {
                    da[i] = probes[i % p];
                    db[i] = probes[(i + shift) % p];
                }
                let got = V::new(da).abs_diff(V::new(db));

                for i in 0..N {
                    assert_eq!(
                        got.as_slice()[i],
                        da[i].abs_diff(db[i]),
                        "abs_diff a={} b={} lane={}",
                        da[i],
                        db[i],
                        i
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
            check!(u8x16, thermite::simd::u8x16<$backend>, u8, 16);
            check!(u16x8, thermite::simd::u16x8<$backend>, u16, 8);
            check!(u16x16, thermite::simd::u16x16<$backend>, u16, 16);
            check!(u32x4, thermite::simd::u32x4<$backend>, u32, 4);
            check!(u32x8, thermite::simd::u32x8<$backend>, u32, 8);
            check!(u64x2, thermite::simd::u64x2<$backend>, u64, 2);
            check!(u64x4, thermite::simd::u64x4<$backend>, u64, 4);
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
