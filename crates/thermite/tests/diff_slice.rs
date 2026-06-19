//! Coverage for SIMD slice iteration: the `SimdSlice` extension trait
//! (`slice.rs`), the `Unaligned`/`UnalignedMut` iterators (`vector/unaligned.rs`),
//! and the `StreamingVector`/`StreamingVectorMut` handles (`vector/streaming.rs`).
//! All three were at 0% coverage.
//!
//! Strategy: build `[elem]` filled with `0..n`, run each iteration strategy, and
//! check the result reconstructs (or transforms) the slice exactly. Lengths span
//! the interesting cases: empty, sub-lane, exact multiples, and odd remainders.
//! Values `0..257` and `±1` are exactly representable in f32/i32, so equality is
//! exact.
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use thermite::Vector;
use thermite::prelude::*;
use thermite::simd::Simd;

use thermite::backend::scalar::Scalar;
use thermite::backend::x86_v1::X86V1;
use thermite::backend::x86_v2::X86V2;
use thermite::backend::x86_v3::X86V3;

macro_rules! slice_suite {
    ($mod:ident, $backend:ty, $reg:ident, $elem:ty, $bl:expr) => {
        mod $mod {
            use super::*;
            type V = Vector<<$backend as Simd>::$reg>;
            const LANES: usize = <V as GenericVector>::LANES;

            fn lengths() -> Vec<usize> {
                let l = LANES;
                vec![0, 1, l - 1, l, l + 1, 2 * l, 2 * l + 3, 5 * l + 1, 257]
            }
            fn fill(n: usize) -> Vec<$elem> {
                (0..n).map(|i| i as $elem).collect()
            }
            /// Aligned-middle bounds `[hl, end)` of `data`, as found by `align_slice`
            /// (the region with no leading/trailing scalar remainder). Must be computed
            /// from the *same* allocation it will index - a different `Vec` of equal
            /// length can have different pointer alignment.
            fn mid_bounds(data: &[$elem]) -> (usize, usize) {
                let (head, _chunks, tail) = data.try_aligned_simd_iter::<V>();
                (head.len(), data.len() - tail.len())
            }

            #[test]
            fn try_aligned_shared() {
                for &n in &lengths() {
                    let data = fill(n);
                    let (head, chunks, tail) = data.try_aligned_simd_iter::<V>();
                    assert!(head.len() < LANES, concat!($bl, " head >= LANES"));
                    assert!(tail.len() < LANES, concat!($bl, " tail >= LANES"));
                    let mut recon: Vec<$elem> = head.to_vec();
                    for c in chunks {
                        recon.extend_from_slice(c.as_slice());
                    }
                    recon.extend_from_slice(tail);
                    assert_eq!(recon, data, concat!($bl, " try_aligned reconstruct, n={}"), n);
                }
            }

            #[test]
            fn aligned_and_streaming_shared() {
                for &n in &lengths() {
                    let data = fill(n);
                    let (hl, end) = mid_bounds(&data);
                    let mid = &data[hl..end];

                    // aligned_simd_iter over the exactly-aligned middle (no panic).
                    let mut recon = Vec::new();
                    for c in mid.aligned_simd_iter::<V>() {
                        recon.extend_from_slice(c.as_slice());
                    }
                    assert_eq!(recon.as_slice(), mid, concat!($bl, " aligned_simd_iter, n={}"), n);

                    // streaming_simd_iter: NT load must equal cached load.
                    let mut recon2 = Vec::new();
                    for sv in mid.streaming_simd_iter::<V>() {
                        let nt = sv.load();
                        let cached = sv.load_cached();
                        assert_eq!(
                            nt.as_slice(),
                            cached.as_slice(),
                            concat!($bl, " NT vs cached load")
                        );
                        recon2.extend_from_slice(nt.as_slice());
                    }
                    assert_eq!(recon2.as_slice(), mid, concat!($bl, " streaming load, n={}"), n);
                }
            }

            #[test]
            fn aligned_mut_and_streaming_mut() {
                for &n in &lengths() {
                    let mut data = fill(n);
                    let (hl, end) = mid_bounds(&data);
                    let mut expected = data.clone();

                    // try_aligned_simd_iter_mut: +1 over the aligned middle.
                    for e in &mut expected[hl..end] {
                        *e = *e + 1 as $elem;
                    }
                    {
                        let (_h, chunks, _t) = data.try_aligned_simd_iter_mut::<V>();
                        for c in chunks {
                            *c = *c + V::ONE;
                        }
                    }
                    assert_eq!(data, expected, concat!($bl, " try_aligned_mut +1, n={}"), n);

                    // streaming_simd_iter_mut: +1 again (NT store), plus a store_cached round-trip.
                    for e in &mut expected[hl..end] {
                        *e = *e + 1 as $elem;
                    }
                    {
                        let mid = &mut data[hl..end];
                        for mut sv in mid.streaming_simd_iter_mut::<V>() {
                            let v = sv.load();
                            sv.store_cached(v); // no-op round-trip through the cached path
                            sv.store(v + V::ONE);
                        }
                    }
                    assert_eq!(data, expected, concat!($bl, " streaming_mut +1, n={}"), n);

                    // aligned_simd_iter_mut: -1 to restore the aligned middle.
                    for e in &mut expected[hl..end] {
                        *e = *e - 1 as $elem;
                    }
                    {
                        let mid = &mut data[hl..end];
                        for c in mid.aligned_simd_iter_mut::<V>() {
                            *c = *c - V::ONE;
                        }
                    }
                    assert_eq!(data, expected, concat!($bl, " aligned_mut -1, n={}"), n);
                }
            }

            #[test]
            fn unaligned_shared() {
                for &n in &lengths() {
                    let data = fill(n);
                    let (it, rem) = data.unaligned_simd_iter::<V>();
                    assert!(rem.len() < LANES, concat!($bl, " unaligned rem >= LANES"));
                    assert_eq!(it.len(), n / LANES, concat!($bl, " Unaligned::len"));
                    assert_eq!(it.count(), n / LANES, concat!($bl, " Unaligned::count"));

                    // forward reconstruct
                    let (itf, rem) = data.unaligned_simd_iter::<V>();
                    let mut recon = Vec::new();
                    for v in itf {
                        recon.extend_from_slice(v.as_slice());
                    }
                    recon.extend_from_slice(rem);
                    assert_eq!(recon, data, concat!($bl, " unaligned reconstruct, n={}"), n);

                    // random access: read(i), and out-of-bounds -> None
                    let (itr, _) = data.unaligned_simd_iter::<V>();
                    for i in 0..itr.len() {
                        let v = itr.read(i).unwrap();
                        assert_eq!(
                            v.as_slice(),
                            &data[i * LANES..(i + 1) * LANES],
                            concat!($bl, " read({})"),
                            i
                        );
                    }
                    assert!(
                        itr.read(itr.len()).is_none(),
                        concat!($bl, " read OOB should be None")
                    );
                    assert_eq!(
                        itr.as_slice(),
                        &data[..(n / LANES) * LANES],
                        concat!($bl, " as_slice")
                    );

                    // DoubleEnded: reverse then un-reverse equals forward.
                    let (fa, _) = data.unaligned_simd_iter::<V>();
                    let fwd: Vec<Vec<$elem>> = fa.map(|v| v.as_slice().to_vec()).collect();
                    let (ba, _) = data.unaligned_simd_iter::<V>();
                    let mut bwd: Vec<Vec<$elem>> = ba.rev().map(|v| v.as_slice().to_vec()).collect();
                    bwd.reverse();
                    assert_eq!(fwd, bwd, concat!($bl, " DoubleEnded, n={}"), n);

                    // fold (custom impl) counts the chunks
                    let (fc, _) = data.unaligned_simd_iter::<V>();
                    assert_eq!(
                        fc.fold(0usize, |acc, _| acc + 1),
                        n / LANES,
                        concat!($bl, " fold")
                    );

                    // try_aligned succeeds on the exactly-aligned middle
                    let (hl, endb) = mid_bounds(&data);
                    let mid = &data[hl..endb];
                    let (mit, mrem) = mid.unaligned_simd_iter::<V>();
                    assert_eq!(mrem.len(), 0, concat!($bl, " aligned middle has no remainder"));
                    // try_aligned is only meaningful on a non-empty middle: an empty
                    // slice's dangling pointer is element-aligned, not vector-aligned.
                    if endb > hl {
                        let aligned = mit
                            .try_aligned()
                            .expect(concat!($bl, " try_aligned should be Some on aligned middle"));
                        assert_eq!(
                            aligned.len(),
                            (endb - hl) / LANES,
                            concat!($bl, " try_aligned len")
                        );
                    }
                }
            }

            #[test]
            fn unaligned_mut() {
                for &n in &lengths() {
                    let mut data = fill(n);
                    let full = (n / LANES) * LANES;
                    let mut expected = data.clone();
                    for e in &mut expected[..full] {
                        *e = *e + 1 as $elem;
                    }
                    {
                        let (mut it, _rem) = data.unaligned_simd_iter_mut::<V>();
                        let cnt = it.len(); // via Deref to Unaligned
                        for i in 0..cnt {
                            let v = it.read(i).unwrap(); // via Deref
                            assert!(it.write(i, v + V::ONE), concat!($bl, " write in-bounds"));
                        }
                        assert!(
                            !it.write(cnt, V::ZERO),
                            concat!($bl, " write OOB should be false")
                        );
                    }
                    assert_eq!(data, expected, concat!($bl, " unaligned_mut +1, n={}"), n);
                }
            }

            #[test]
            #[should_panic]
            fn aligned_panics_on_remainder() {
                // 2*LANES+1 is never a multiple of LANES, so there is always a
                // trailing scalar remainder and `aligned_simd_iter` must panic.
                let data = fill(2 * LANES + 1);
                let _ = data.aligned_simd_iter::<V>().count();
            }
        }
    };
}

slice_suite!(v3_f32, X86V3, f32x8, f32, "x86_v3 f32x8");
slice_suite!(v3_i32, X86V3, i32x8, i32, "x86_v3 i32x8");
slice_suite!(v3_f64, X86V3, f64x4, f64, "x86_v3 f64x4");
slice_suite!(v3_i64, X86V3, i64x4, i64, "x86_v3 i64x4");
slice_suite!(v3_u64, X86V3, u64x4, u64, "x86_v3 u64x4");
slice_suite!(v2_f32, X86V2, f32x4, f32, "x86_v2 f32x4");
slice_suite!(v2_f64, X86V2, f64x2, f64, "x86_v2 f64x2");
slice_suite!(v2_i64, X86V2, i64x2, i64, "x86_v2 i64x2");
slice_suite!(v2_u64, X86V2, u64x2, u64, "x86_v2 u64x2");
slice_suite!(v1_f32, X86V1, f32x4, f32, "x86_v1 f32x4");
slice_suite!(v1_f64, X86V1, f64x2, f64, "x86_v1 f64x2");
slice_suite!(v1_i64, X86V1, i64x2, i64, "x86_v1 i64x2");
slice_suite!(v1_u64, X86V1, u64x2, u64, "x86_v1 u64x2");
slice_suite!(scalar_f32, Scalar, f32x4, f32, "scalar f32x4");
