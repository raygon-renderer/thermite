//! `IntegerVector::align` (two-register `palignr`-style element align).
//!
//! `a.align::<OFFSET>(b)` is the window of `LANES` lanes starting at lane
//! `OFFSET` of the concatenation `[a, b]`. With `a[i] = i + 1` and
//! `b[i] = N + i + 1` the concatenation is `concat[k] = k + 1`, so the result
//! must be `got[i] = i + OFFSET + 1` - a clean oracle covering `OFFSET == 0`
//! (returns `a`), `OFFSET == LANES` (returns `b`), and every spill in between.
//! Exercised across the scalar backend and, on x86, v1/v2/v3.

use thermite::backend::scalar::Scalar;
use thermite::prelude::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use thermite::backend::{x86_v1::X86V1, x86_v2::X86V2, x86_v3::X86V3};

macro_rules! check {
    ($name:ident, $vty:ty, $elem:ty, $n:literal, [$($off:literal),*]) => {
        #[test]
        fn $name() {
            type V = $vty;
            const N: usize = $n;

            // Distinct nonzero values so a misplaced/dropped lane is visible;
            // concat[k] == k + 1.
            let mut da = [0 as $elem; N];
            let mut db = [0 as $elem; N];
            for i in 0..N {
                da[i] = (i + 1) as $elem;
                db[i] = (N + i + 1) as $elem;
            }
            let a = V::new(da);
            let b = V::new(db);

            $({
                const OFF: usize = $off;
                let got = a.align::<OFF>(b);
                for i in 0..N {
                    assert_eq!(
                        got.as_slice()[i],
                        (i + OFF + 1) as $elem,
                        "align::<{}> width={} lane={}",
                        OFF, N, i
                    );
                }
            })*
        }
    };
}

macro_rules! suite {
    ($modname:ident, $backend:ty) => {
        mod $modname {
            use super::*;
            check!(
                u8x16,
                thermite::simd::u8x16<$backend>,
                u8,
                16,
                [0, 1, 2, 7, 8, 15, 16]
            );
            check!(
                i8x16,
                thermite::simd::i8x16<$backend>,
                i8,
                16,
                [0, 1, 2, 7, 8, 15, 16]
            );
            check!(u16x8, thermite::simd::u16x8<$backend>, u16, 8, [0, 1, 4, 7, 8]);
            check!(
                u16x16,
                thermite::simd::u16x16<$backend>,
                u16,
                16,
                [0, 1, 8, 15, 16]
            );
            check!(u32x4, thermite::simd::u32x4<$backend>, u32, 4, [0, 1, 2, 3, 4]);
            check!(u32x8, thermite::simd::u32x8<$backend>, u32, 8, [0, 1, 5, 8]);
            check!(i16x8, thermite::simd::i16x8<$backend>, i16, 8, [0, 1, 4, 8]);
            check!(i32x4, thermite::simd::i32x4<$backend>, i32, 4, [0, 2, 4]);
            check!(u64x2, thermite::simd::u64x2<$backend>, u64, 2, [0, 1, 2]);
            // 256-bit on v3 (native), ArrayRegister on v1/v2, 1-lane-chunk array on scalar.
            check!(
                i16x16,
                thermite::simd::i16x16<$backend>,
                i16,
                16,
                [0, 1, 8, 15, 16]
            );
            check!(i32x8, thermite::simd::i32x8<$backend>, i32, 8, [0, 1, 4, 7, 8]);
            check!(i64x4, thermite::simd::i64x4<$backend>, i64, 4, [0, 1, 2, 3, 4]);
            check!(u64x4, thermite::simd::u64x4<$backend>, u64, 4, [0, 1, 2, 4]);
            // Float widths: align is on GenericVector now, so it works for any
            // element type. f32x8/f64x4 are native 256 on v3, ArrayRegister on
            // v1/v2, 1-lane-chunk array on scalar. Values are exact integers, so
            // the lane-movement result compares exactly.
            check!(f32x8, thermite::simd::f32x8<$backend>, f32, 8, [0, 1, 5, 8]);
            check!(f64x4, thermite::simd::f64x4<$backend>, f64, 4, [0, 1, 2, 4]);
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

// Native 256-bit byte align on v3 (AVX2): `i8xN`/`u8xN` are 32 lanes here, so
// `ob == OFFSET` and this exercises every arm of the 33-way match - including the
// odd byte offsets that the wider element types (EB >= 2) never reach.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod v3_native_bytes {
    use super::*;
    check!(
        i8xN,
        thermite::simd::i8xN<X86V3>,
        i8,
        32,
        [0, 1, 2, 3, 15, 16, 17, 18, 31, 32]
    );
    check!(
        u8xN,
        thermite::simd::u8xN<X86V3>,
        u8,
        32,
        [0, 1, 2, 3, 15, 16, 17, 18, 31, 32]
    );
}
