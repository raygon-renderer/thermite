//! Coverage for SIMD slice iteration: the `SimdSlice` extension trait
//! (`slice.rs`), the `Unaligned`/`UnalignedMut` iterators (`vector/unaligned.rs`),
//! and the `StreamingVector`/`StreamingVectorMut` handles (`vector/streaming.rs`).
//!
//! Strategy: build `[elem]` filled with `0..n`, run each iteration strategy, and
//! check the result reconstructs (or transforms) the slice exactly. Lengths span
//! the interesting cases: empty, sub-lane, exact multiples, and odd remainders.
//! Values `0..257` and `+-1` are exactly representable in f32/i32, so equality is
//! exact. Every slot below runs on every backend (a slot is native on some and
//! `ArrayRegister`-emulated on others, and the iterators must not care).
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use thermite::Vector;
use thermite::prelude::*;
use thermite::simd::Simd;

/// Per-slot context: the vector type, its lane count and the three helpers.
/// Emitted as items at the top of each test body (the concrete stamper allows
/// `type` items because `S` is an alias there).
macro_rules! ctx {
    ($reg:ident, $elem:ty) => {
        type V = Vector<<S as Simd>::$reg>;
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
        /// from the _same_ allocation it will index, since a different `Vec` of equal
        /// length can have different pointer alignment.
        fn mid_bounds(data: &[$elem]) -> (usize, usize) {
            let (head, _chunks, tail) = data.try_aligned_simd_iter::<V>();
            (head.len(), data.len() - tail.len())
        }
    };
}

macro_rules! try_aligned_shared {
    ($reg:ident, $elem:ty) => {{
        ctx!($reg, $elem);
        for &n in &lengths() {
            let data = fill(n);
            let (head, chunks, tail) = data.try_aligned_simd_iter::<V>();
            assert!(head.len() < LANES, "head >= LANES");
            assert!(tail.len() < LANES, "tail >= LANES");
            let mut recon: Vec<$elem> = head.to_vec();
            for c in chunks {
                recon.extend_from_slice(c.as_slice());
            }
            recon.extend_from_slice(tail);
            assert_eq!(recon, data, "try_aligned reconstruct, n={}", n);
        }
    }};
}

macro_rules! aligned_and_streaming_shared {
    ($reg:ident, $elem:ty) => {{
        ctx!($reg, $elem);
        for &n in &lengths() {
            let data = fill(n);
            let (hl, end) = mid_bounds(&data);
            let mid = &data[hl..end];

            let mut recon = Vec::new();
            for c in mid.aligned_simd_iter::<V>() {
                recon.extend_from_slice(c.as_slice());
            }
            assert_eq!(recon.as_slice(), mid, "aligned_simd_iter, n={}", n);

            let mut recon2 = Vec::new();
            for sv in mid.streaming_simd_iter::<V>() {
                let nt = sv.load();
                let cached = sv.load_cached();
                assert_eq!(nt.as_slice(), cached.as_slice(), "NT vs cached load");
                recon2.extend_from_slice(nt.as_slice());
            }
            assert_eq!(recon2.as_slice(), mid, "streaming load, n={}", n);
        }
    }};
}

macro_rules! aligned_mut_and_streaming_mut {
    ($reg:ident, $elem:ty) => {{
        ctx!($reg, $elem);
        for &n in &lengths() {
            let mut data = fill(n);
            let (hl, end) = mid_bounds(&data);
            let mut expected = data.clone();

            for e in &mut expected[hl..end] {
                *e = *e + 1 as $elem;
            }
            {
                let (_h, chunks, _t) = data.try_aligned_simd_iter_mut::<V>();
                for c in chunks {
                    *c = *c + V::ONE;
                }
            }
            assert_eq!(data, expected, "try_aligned_mut +1, n={}", n);

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
            assert_eq!(data, expected, "streaming_mut +1, n={}", n);

            for e in &mut expected[hl..end] {
                *e = *e - 1 as $elem;
            }
            {
                let mid = &mut data[hl..end];
                for c in mid.aligned_simd_iter_mut::<V>() {
                    *c = *c - V::ONE;
                }
            }
            assert_eq!(data, expected, "aligned_mut -1, n={}", n);
        }
    }};
}

macro_rules! unaligned_shared {
    ($reg:ident, $elem:ty) => {{
        ctx!($reg, $elem);
        for &n in &lengths() {
            let data = fill(n);
            let (it, rem) = data.unaligned_simd_iter::<V>();
            assert!(rem.len() < LANES, "unaligned rem >= LANES");
            assert_eq!(it.len(), n / LANES, "Unaligned::len");
            assert_eq!(it.count(), n / LANES, "Unaligned::count");

            let (itf, rem) = data.unaligned_simd_iter::<V>();
            let mut recon = Vec::new();
            for v in itf {
                recon.extend_from_slice(v.as_slice());
            }
            recon.extend_from_slice(rem);
            assert_eq!(recon, data, "unaligned reconstruct, n={}", n);

            let (itr, _) = data.unaligned_simd_iter::<V>();
            for i in 0..itr.len() {
                let v = itr.read(i).unwrap();
                assert_eq!(v.as_slice(), &data[i * LANES..(i + 1) * LANES], "read({})", i);
            }
            assert!(itr.read(itr.len()).is_none(), "read OOB should be None");
            assert_eq!(itr.as_slice(), &data[..(n / LANES) * LANES], "as_slice");

            let (fa, _) = data.unaligned_simd_iter::<V>();
            let fwd: Vec<Vec<$elem>> = fa.map(|v| v.as_slice().to_vec()).collect();
            let (ba, _) = data.unaligned_simd_iter::<V>();
            let mut bwd: Vec<Vec<$elem>> = ba.rev().map(|v| v.as_slice().to_vec()).collect();
            bwd.reverse();
            assert_eq!(fwd, bwd, "DoubleEnded, n={}", n);

            let (fc, _) = data.unaligned_simd_iter::<V>();
            assert_eq!(fc.fold(0usize, |acc, _| acc + 1), n / LANES, "fold");

            let (hl, endb) = mid_bounds(&data);
            let mid = &data[hl..endb];
            let (mit, mrem) = mid.unaligned_simd_iter::<V>();
            assert_eq!(mrem.len(), 0, "aligned middle has no remainder");
            if endb > hl {
                let aligned = mit.try_aligned().expect("try_aligned should be Some on aligned middle");
                assert_eq!(aligned.len(), (endb - hl) / LANES, "try_aligned len");
            }
        }
    }};
}

macro_rules! unaligned_mut {
    ($reg:ident, $elem:ty) => {{
        ctx!($reg, $elem);
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
                    assert!(it.write(i, v + V::ONE), "write in-bounds");
                }
                assert!(!it.write(cnt, V::ZERO), "write OOB should be false");
            }
            assert_eq!(data, expected, "unaligned_mut +1, n={}", n);
        }
    }};
}

/// Every slot, for one of the per-strategy macros above.
macro_rules! all_slots {
    ($m:ident) => {{
        $m!(f32x4, f32);
        $m!(f32x8, f32);
        $m!(f32x16, f32);
        $m!(f64x2, f64);
        $m!(f64x4, f64);
        $m!(i32x8, i32);
        $m!(i64x2, i64);
        $m!(i64x4, i64);
        $m!(u64x2, u64);
        $m!(u64x4, u64);
    }};
}

for_each_backend_concrete! {
    fn try_aligned_shared() { all_slots!(try_aligned_shared) }
    fn aligned_and_streaming_shared() { all_slots!(aligned_and_streaming_shared) }
    fn aligned_mut_and_streaming_mut() { all_slots!(aligned_mut_and_streaming_mut) }
    fn unaligned_shared() { all_slots!(unaligned_shared) }
    fn unaligned_mut() { all_slots!(unaligned_mut) }

    #[should_panic]
    fn aligned_panics_on_remainder() {
        ctx!(f32x8, f32);
        let data = fill(2 * LANES + 1);
        let _ = data.aligned_simd_iter::<V>().count();
    }
}
