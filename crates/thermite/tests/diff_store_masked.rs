//! Differential coverage for `store_masked` across every x86_v3 register that
//! overrides it with `vmaskmov` (`_mm*_maskstore_{ps,pd,epi32,epi64}`), against
//! the scalar lane-walking default in `Register::store_masked`.
//!
//! Exhausts all `2^LANES` mask patterns per register and checks three things:
//! masked-on lanes take the vector's value, masked-off lanes keep their prior
//! contents, and nothing past `LANES` is touched (the buffer is deliberately
//! wider than the widest register under test).
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use thermite::Vector;
use thermite::prelude::*;
use thermite::simd::Simd;

/// 32-byte aligned scratch, sized for the widest register plus a tail guard.
#[repr(align(32))]
struct Aligned<T>([T; 16]);

macro_rules! diff_store_masked {
    ($name:ident, $reg:ty, $elem:ty) => {
        #[test]
        fn $name() {
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
        }
    };
}

mod x86_v3 {
    use super::*;
    use thermite::backend::x86_v3::X86V3;

    diff_store_masked!(f32x4, <X86V3 as Simd>::f32x4, f32);
    diff_store_masked!(f32x8, <X86V3 as Simd>::f32x8, f32);
    diff_store_masked!(f64x2, <X86V3 as Simd>::f64x2, f64);
    diff_store_masked!(f64x4, <X86V3 as Simd>::f64x4, f64);

    diff_store_masked!(i32x4, <X86V3 as Simd>::i32x4, i32);
    diff_store_masked!(i32x8, <X86V3 as Simd>::i32x8, i32);
    diff_store_masked!(i64x2, <X86V3 as Simd>::i64x2, i64);
    diff_store_masked!(i64x4, <X86V3 as Simd>::i64x4, i64);

    diff_store_masked!(u32x4, <X86V3 as Simd>::u32x4, u32);
    diff_store_masked!(u32x8, <X86V3 as Simd>::u32x8, u32);
    diff_store_masked!(u64x2, <X86V3 as Simd>::u64x2, u64);
    diff_store_masked!(u64x4, <X86V3 as Simd>::u64x4, u64);
}

/// The scalar backend keeps the lane-walking default; same contract must hold.
mod scalar {
    use super::*;
    use thermite::backend::scalar::Scalar;

    diff_store_masked!(f32x4, <Scalar as Simd>::f32x4, f32);
    diff_store_masked!(f64x4, <Scalar as Simd>::f64x4, f64);
    diff_store_masked!(i32x8, <Scalar as Simd>::i32x8, i32);
    diff_store_masked!(u64x2, <Scalar as Simd>::u64x2, u64);
}
