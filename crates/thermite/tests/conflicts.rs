//! `IntegerVector::count_conflicts` and `PartialOrdVector::group_by_value`.
//!
//! Both are about duplicate values across lanes, so both are tested by
//! enumerating *every* value pattern over a small alphabet - that makes
//! duplicates dense, which is exactly the interesting case and is what a random
//! sweep would mostly miss. Oracles are plain scalar loops over the input array.
//!
//! Run across the scalar backend and every native backend for the target: the
//! rotate ladder underneath `count_conflicts` is built on `align`, whose
//! lowering differs per backend, so cross-backend agreement is the point.

use thermite::backend::scalar::Scalar;
use thermite::prelude::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use thermite::backend::{x86_v1::X86V1, x86_v2::X86V2, x86_v3::X86V3};

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use thermite::backend::wasm::Wasm;

#[cfg(target_arch = "aarch64")]
use thermite::backend::neon::Neon;

/// Enumerate every `ALPHABET^N` value pattern and check both ops against
/// scalar oracles.
macro_rules! check {
    ($name:ident, $vty:ty, $elem:ty, $n:literal, $alphabet:literal) => {
        #[test]
        fn $name() {
            type V = $vty;
            const N: usize = $n;
            const A: usize = $alphabet;

            // No per-lane accessor on `Mask`; materialize through `select`.
            let set_at = |m: <V as GenericVector>::Mask, i: usize| {
                m.select(V::ONE, V::ZERO).as_slice()[i] != 0 as $elem
            };

            let total = A.pow(N as u32);
            for pattern in 0..total {
                // Decode the pattern into lane values.
                let mut d = [0 as $elem; N];
                let mut p = pattern;
                for i in 0..N {
                    d[i] = (p % A) as $elem;
                    p /= A;
                }
                let v = V::new(d.into());

                // --- count_conflicts -------------------------------------
                let mut want = [0 as $elem; N];
                for i in 0..N {
                    let mut c = 0;
                    for j in 0..i {
                        if d[j] == d[i] {
                            c += 1;
                        }
                    }
                    want[i] = c as $elem;
                }

                let got = v.count_conflicts();
                for i in 0..N {
                    assert_eq!(
                        got.as_slice()[i],
                        want[i],
                        "count_conflicts lane={} input={:?} got={:?}",
                        i,
                        d,
                        got.as_slice()
                    );
                }

                // First-occurrence mask falls out of the count.
                let first = got.cmp_eq(V::ZERO);
                for i in 0..N {
                    let is_first = !d[..i].contains(&d[i]);
                    assert_eq!(set_at(first, i), is_first, "first-occurrence lane={} input={:?}", i, d);
                }

                // --- group_by_value --------------------------------------
                // Exercise a non-trivial `valid` derived from the pattern, so
                // masked-out lanes are covered too.
                let valid_bits = pattern % (1usize << N);
                let mut sel = [0 as $elem; N];
                for i in 0..N {
                    if (valid_bits >> i) & 1 == 1 {
                        sel[i] = 1 as $elem;
                    }
                }
                let valid = V::new(sel.into()).cmp_ne(V::ZERO);

                let mut seen = [false; N];
                let mut groups = v.group_by_value(valid);
                let mut order = std::vec::Vec::new();

                while let Some((value, lanes)) = groups.next_group() {
                    order.push(value);
                    let mut any = false;
                    for i in 0..N {
                        if set_at(lanes, i) {
                            any = true;
                            assert!((valid_bits >> i) & 1 == 1, "group included an invalid lane {}", i);
                            assert_eq!(d[i], value, "group value mismatch lane={} input={:?}", i, d);
                            assert!(!seen[i], "lane {} yielded twice", i);
                            seen[i] = true;
                        }
                    }
                    assert!(any, "empty group yielded");
                }

                // Every valid lane yielded exactly once, invalid lanes never.
                for i in 0..N {
                    assert_eq!(seen[i], (valid_bits >> i) & 1 == 1, "coverage lane={} input={:?}", i, d);
                }
                assert!(groups.is_empty(), "iterator finished with lanes remaining");

                // Groups are distinct and in order of first occurrence.
                for a in 0..order.len() {
                    for b in (a + 1)..order.len() {
                        assert_ne!(order[a], order[b], "duplicate group value");
                    }
                }
                let mut expect_order = std::vec::Vec::new();
                for i in 0..N {
                    if (valid_bits >> i) & 1 == 1 && !expect_order.contains(&d[i]) {
                        expect_order.push(d[i]);
                    }
                }
                assert_eq!(order, expect_order, "group order input={:?}", d);
            }
        }
    };
}

macro_rules! suite {
    ($modname:ident, $backend:ty) => {
        mod $modname {
            use super::*;

            // 4^4 = 256 patterns, 3^8 = 6561, 2^16 = 65536 - all exhaustive.
            check!(i32x4, thermite::simd::i32x4<$backend>, i32, 4, 4);
            check!(u32x4, thermite::simd::u32x4<$backend>, u32, 4, 4);
            check!(i32x8, thermite::simd::i32x8<$backend>, i32, 8, 3);
            check!(u16x8, thermite::simd::u16x8<$backend>, u16, 8, 3);
            check!(i64x4, thermite::simd::i64x4<$backend>, i64, 4, 4);
            check!(u64x2, thermite::simd::u64x2<$backend>, u64, 2, 4);
            check!(i16x16, thermite::simd::i16x16<$backend>, i16, 16, 2);
            check!(u8x16, thermite::simd::u8x16<$backend>, u8, 16, 2);

            /// A uniform packet is one group; a fully-distinct packet is `LANES`
            /// groups of one lane each. The two ends of the divergence range.
            #[test]
            fn degenerate() {
                type V = thermite::simd::i32x8<$backend>;

                let uniform = V::splat(7);
                let mut g = uniform.group_by_value(<V as GenericVector>::Mask::TRUTHY);
                let (val, lanes) = g.next_group().expect("uniform packet has one group");
                assert_eq!(val, 7);
                assert_eq!(lanes.count_set(), 8);
                assert!(g.next_group().is_none());
                assert_eq!(uniform.count_conflicts().into_array(), [0i32, 1, 2, 3, 4, 5, 6, 7].into());

                let distinct = V::indexed();
                assert_eq!(distinct.count_conflicts().into_array(), [0i32; 8].into());
                assert_eq!(distinct.group_by_value(<V as GenericVector>::Mask::TRUTHY).count(), 8);

                // An empty `valid` yields nothing at all.
                let mut none = distinct.group_by_value(<V as GenericVector>::Mask::FALSY);
                assert!(none.is_empty());
                assert!(none.next_group().is_none());
            }
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

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
suite!(wasm, Wasm);
#[cfg(target_arch = "aarch64")]
suite!(neon, Neon);
