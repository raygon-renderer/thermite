//! Differential coverage for `store_masked` on every backend: the x86_v3
//! `vmaskmov` overrides (`_mm*_maskstore_{ps,pd,epi32,epi64}`), the AVX-512
//! opmask stores, and the scalar lane-walking default in `Register::store_masked`.
//!
//! Exhausts all `2^LANES` mask patterns per register and checks three things:
//! masked-on lanes take the vector's value, masked-off lanes keep their prior
//! contents, and nothing past `LANES` is touched (the buffer is deliberately
//! wider than the widest register under test).
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

/// 32-byte aligned scratch, sized for the widest register plus a tail guard.
#[repr(align(32))]
struct Aligned<T>([T; 16]);

macro_rules! store_masked {
    ($reg:ty, $elem:ty) => {{
        type V = Vector<$reg>;

        const L: usize = <V as GenericVector>::LANES;
        assert!(L <= 8, "buffer is 16 wide; a tail guard needs L < 16");

        // Distinct, exactly representable values: stored 1..=L, prior 101...
        let stored: [$elem; 16] = core::array::from_fn(|i| (i + 1) as $elem);
        let prior: [$elem; 16] = core::array::from_fn(|i| (i + 101) as $elem);

        let value = unsafe { V::load_unaligned(stored.as_ptr()) };

        for bits in 0..(1usize << L) {
            let sel: [$elem; 16] = core::array::from_fn(|i| ((bits >> (i % L)) & 1) as $elem);
            let mask = unsafe { V::load_unaligned(sel.as_ptr()) }.cmp_ne(V::ZERO);

            let mut buf = Aligned(prior);
            unsafe { value.store_masked(mask, buf.0.as_mut_ptr()) };

            let expected: [$elem; 16] = core::array::from_fn(|i| {
                if i < L && (bits >> i) & 1 == 1 {
                    stored[i]
                } else {
                    prior[i] // masked-off lanes and the tail past LANES
                }
            });

            assert_eq!(
                buf.0,
                expected,
                "{}: mask bits {bits:#b} over {L} lanes",
                stringify!($reg)
            );
        }
    }};
}

for_each_backend_concrete! {
    fn f32x4() { store_masked!(<S as Simd>::f32x4, f32) }
    fn f32x8() { store_masked!(<S as Simd>::f32x8, f32) }
    fn f64x2() { store_masked!(<S as Simd>::f64x2, f64) }
    fn f64x4() { store_masked!(<S as Simd>::f64x4, f64) }
    fn f64x8() { store_masked!(<S as Simd>::f64x8, f64) }

    fn i32x4() { store_masked!(<S as Simd>::i32x4, i32) }
    fn i32x8() { store_masked!(<S as Simd>::i32x8, i32) }
    fn i64x2() { store_masked!(<S as Simd>::i64x2, i64) }
    fn i64x4() { store_masked!(<S as Simd>::i64x4, i64) }
    fn i64x8() { store_masked!(<S as Simd>::i64x8, i64) }

    fn u32x4() { store_masked!(<S as Simd>::u32x4, u32) }
    fn u32x8() { store_masked!(<S as Simd>::u32x8, u32) }
    fn u64x2() { store_masked!(<S as Simd>::u64x2, u64) }
    fn u64x4() { store_masked!(<S as Simd>::u64x4, u64) }
    fn u64x8() { store_masked!(<S as Simd>::u64x8, u64) }

    fn u16x8() { store_masked!(<S as Simd>::u16x8, u16) }
    fn i8x8() { store_masked!(<S as Simd>::i8x8, i8) }
}
