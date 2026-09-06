//! `NumericVector` inclusive prefix scans (`prefix_sum`/`min`/`max` + reverse forms).
//!
//! Each check computes the answer independently with a plain sequential loop over the
//! input array and compares lane by lane, so the log-depth `align` ladder is validated
//! against the obvious definition rather than against itself. Run across the scalar
//! backend and every native backend for the target, which is what makes this a real
//! test of the ladder: the scalar backend has `HAS_NATIVE_ALIGN == false` and so takes
//! the sequential fallback, while v1/v2/v3/v4/wasm/neon take the ladder, and the two
//! must agree.
//!
//! Values are small exact integers (also when the element type is float) because the
//! ladder reassociates the sum: `prefix_sum` on a tree is not bit-identical to a
//! sequential float sum for arbitrary inputs. So the test pins down lane routing
//! rather than float summation order.
//!
//! `infinities` covers the case that dictated the fill-value design: a `+inf` lane must
//! survive `prefix_min` (it would not if the shifted-in fill were `MAX` rather than a
//! broadcast of the edge lane).
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

        // Small (no-overflow even at i8 x 32) and non-monotonic, so a prefix
        // min/max actually changes across lanes and a swapped lane is visible.
        let mut d = [0 as $elem; N];
        for i in 0..N {
            d[i] = (1 + ((i * 37 + 11) % 23)) as $elem;
        }
        let v = V::new(d);

        let mut want_sum = [0 as $elem; N];
        let mut want_min = [0 as $elem; N];
        let mut want_max = [0 as $elem; N];
        for i in 0..N {
            want_sum[i] = if i == 0 { d[0] } else { want_sum[i - 1] + d[i] };
            want_min[i] = if i == 0 {
                d[0]
            } else if d[i] < want_min[i - 1] {
                d[i]
            } else {
                want_min[i - 1]
            };
            want_max[i] = if i == 0 {
                d[0]
            } else if d[i] > want_max[i - 1] {
                d[i]
            } else {
                want_max[i - 1]
            };
        }

        let mut want_rsum = [0 as $elem; N];
        let mut want_rmin = [0 as $elem; N];
        let mut want_rmax = [0 as $elem; N];
        for i in (0..N).rev() {
            want_rsum[i] = if i == N - 1 { d[i] } else { want_rsum[i + 1] + d[i] };
            want_rmin[i] = if i == N - 1 {
                d[i]
            } else if d[i] < want_rmin[i + 1] {
                d[i]
            } else {
                want_rmin[i + 1]
            };
            want_rmax[i] = if i == N - 1 {
                d[i]
            } else if d[i] > want_rmax[i + 1] {
                d[i]
            } else {
                want_rmax[i + 1]
            };
        }

        let cases: [(&str, V, &[$elem; N]); 6] = [
            ("prefix_sum", v.prefix_sum(), &want_sum),
            ("prefix_min", v.prefix_min(), &want_min),
            ("prefix_max", v.prefix_max(), &want_max),
            ("reverse_prefix_sum", v.reverse_prefix_sum(), &want_rsum),
            ("reverse_prefix_min", v.reverse_prefix_min(), &want_rmin),
            ("reverse_prefix_max", v.reverse_prefix_max(), &want_rmax),
        ];

        for (label, got, want) in cases {
            for i in 0..N {
                assert_eq!(
                    got.as_slice()[i],
                    want[i],
                    "{} width={} lane={} input={:?}",
                    label,
                    N,
                    i,
                    d
                );
            }
        }
    }};
}

for_each_backend_concrete! {
    fn i32x4() { check!(thermite::simd::i32x4<S>, i32, 4) }
    fn i32x8() { check!(thermite::simd::i32x8<S>, i32, 8) }
    fn i32x16() { check!(thermite::simd::i32x16<S>, i32, 16) }
    fn u32x4() { check!(thermite::simd::u32x4<S>, u32, 4) }
    fn u32x8() { check!(thermite::simd::u32x8<S>, u32, 8) }
    fn u32x16() { check!(thermite::simd::u32x16<S>, u32, 16) }
    fn u16x8() { check!(thermite::simd::u16x8<S>, u16, 8) }
    fn i16x16() { check!(thermite::simd::i16x16<S>, i16, 16) }
    fn u64x2() { check!(thermite::simd::u64x2<S>, u64, 2) }
    fn i64x4() { check!(thermite::simd::i64x4<S>, i64, 4) }
    fn u64x8() { check!(thermite::simd::u64x8<S>, u64, 8) }
    fn u8x16() { check!(thermite::simd::u8x16<S>, u8, 16) }
    fn f32x4() { check!(thermite::simd::f32x4<S>, f32, 4) }
    fn f32x8() { check!(thermite::simd::f32x8<S>, f32, 8) }
    fn f32x16() { check!(thermite::simd::f32x16<S>, f32, 16) }
    fn f64x2() { check!(thermite::simd::f64x2<S>, f64, 2) }
    fn f64x4() { check!(thermite::simd::f64x4<S>, f64, 4) }
    fn f64x8() { check!(thermite::simd::f64x8<S>, f64, 8) }

    fn infinities() {
        type V = thermite::simd::f32x4<S>;

        let v = V::new([f32::INFINITY, 1.0, 2.0, 3.0]);
        assert_eq!(
            v.prefix_min().into_array().as_slice(),
            [f32::INFINITY, 1.0, 1.0, 1.0].as_slice(),
            "prefix_min must not clamp +inf to f32::MAX"
        );

        let w = V::new([f32::NEG_INFINITY, -1.0, -2.0, -3.0]);
        assert_eq!(
            w.prefix_max().into_array().as_slice(),
            [f32::NEG_INFINITY, -1.0, -1.0, -1.0].as_slice(),
            "prefix_max must not clamp -inf to f32::MIN"
        );

        let r = V::new([3.0, 2.0, 1.0, f32::INFINITY]);
        assert_eq!(
            r.reverse_prefix_min().into_array().as_slice(),
            [1.0, 1.0, 1.0, f32::INFINITY].as_slice(),
            "reverse_prefix_min must not clamp +inf to f32::MAX"
        );
    }

    fn degenerate() {
        type V = thermite::simd::i32x8<S>;

        let c = V::splat(7);
        assert_eq!(c.prefix_min().into_array(), [7i32; 8].into());
        assert_eq!(c.prefix_max().into_array(), [7i32; 8].into());
        assert_eq!(c.reverse_prefix_min().into_array(), [7i32; 8].into());
        assert_eq!(c.reverse_prefix_max().into_array(), [7i32; 8].into());

        let ones = V::splat(1);
        assert_eq!(ones.prefix_sum().into_array(), [1i32, 2, 3, 4, 5, 6, 7, 8].into());
        assert_eq!(
            ones.reverse_prefix_sum().into_array(),
            [8i32, 7, 6, 5, 4, 3, 2, 1].into()
        );
    }

    fn last_lane_primitives() {
        {
            type V = thermite::simd::i32x8<S>;
            let mut d = [0i32; 8];
            for (i, v) in d.iter_mut().enumerate() {
                *v = (i as i32) * 3 - 5;
            }
            let v = V::new(d);
            assert_eq!(v.first_element(), d[0]);
            assert_eq!(v.last_element(), d[7]);
            assert_eq!(V::splat(v.last_element()).into_array(), [d[7]; 8].into());
        }
        {
            type V = thermite::simd::u32x16<S>;
            let mut d = [0u32; 16];
            for (i, v) in d.iter_mut().enumerate() {
                *v = (i as u32) * 7 + 3;
            }
            let v = V::new(d);
            assert_eq!(v.first_element(), d[0]);
            assert_eq!(v.last_element(), d[15]);
            assert_eq!(V::splat(v.last_element()).into_array(), [d[15]; 16].into());
        }
        {
            type V = thermite::simd::u32x2<S>;
            let v = V::new([11u32, 22]);
            assert_eq!(v.first_element(), 11);
            assert_eq!(v.last_element(), 22);
            assert_eq!(V::splat(v.last_element()).into_array(), [22u32; 2].into());
        }
    }
}
