//! `UnsignedIntegerVector::abs_diff` correctness.
//!
//! `a.abs_diff(b)` is the per-lane unsigned `|a - b|`, computed branchlessly as
//! `(a -| b) | (b -| a)` with saturating subtraction. Verified against Rust's
//! `uN::abs_diff` oracle over all ordered pairs of a probe set (covering equal,
//! a<b, a>b, and the 0/MAX corners). The default composes `saturating_sub`,
//! which is natively overridden per backend, so this runs on every backend.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use thermite::prelude::*;

macro_rules! check {
    ($vty:ty, $elem:ty, $n:literal) => {{
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
    }};
}

for_each_backend_concrete! {
    fn u8x16() { check!(thermite::simd::u8x16<S>, u8, 16) }
    fn u16x8() { check!(thermite::simd::u16x8<S>, u16, 8) }
    fn u16x16() { check!(thermite::simd::u16x16<S>, u16, 16) }
    fn u32x4() { check!(thermite::simd::u32x4<S>, u32, 4) }
    fn u32x8() { check!(thermite::simd::u32x8<S>, u32, 8) }
    fn u32x16() { check!(thermite::simd::u32x16<S>, u32, 16) }
    fn u64x2() { check!(thermite::simd::u64x2<S>, u64, 2) }
    fn u64x4() { check!(thermite::simd::u64x4<S>, u64, 4) }
    fn u64x8() { check!(thermite::simd::u64x8<S>, u64, 8) }
}
