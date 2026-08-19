//! `NumericVector` inclusive prefix scans (`prefix_sum`/`min`/`max` + reverse forms).
//!
//! Each check computes the answer independently with a plain sequential loop over the
//! input array and compares lane by lane, so the log-depth `align` ladder is validated
//! against the obvious definition rather than against itself. Run across the scalar
//! backend and every native backend for the target, which is what makes this a real
//! test of the ladder: the scalar backend has `HAS_NATIVE_ALIGN == false` and so takes
//! the sequential fallback, while v1/v2/v3/wasm/neon take the ladder, and the two must
//! agree.
//!
//! Values are small exact integers (also when the element type is float) because the
//! ladder reassociates the sum: `prefix_sum` on a tree is not bit-identical to a
//! sequential float sum for arbitrary inputs. So the test pins down lane routing
//! rather than float summation order.
//!
//! `infinities` covers the case that dictated the fill-value design: a `+inf` lane must
//! survive `prefix_min` (it would not if the shifted-in fill were `MAX` rather than a
//! broadcast of the edge lane).

use thermite::backend::scalar::Scalar;
use thermite::prelude::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use thermite::backend::{x86_v1::X86V1, x86_v2::X86V2, x86_v3::X86V3};

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use thermite::backend::wasm::Wasm;

#[cfg(target_arch = "aarch64")]
use thermite::backend::neon::Neon;

macro_rules! check {
    ($name:ident, $vty:ty, $elem:ty, $n:literal) => {
        #[test]
        fn $name() {
            type V = $vty;
            const N: usize = $n;

            // Small (no-overflow even at i8 x 32) and non-monotonic, so a prefix
            // min/max actually changes across lanes and a swapped lane is visible.
            let mut d = [0 as $elem; N];
            for i in 0..N {
                d[i] = (1 + ((i * 37 + 11) % 23)) as $elem;
            }
            let v = V::new(d);

            // --- forward oracles
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

            // --- reverse oracles
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
        }
    };
}

macro_rules! suite {
    ($modname:ident, $backend:ty) => {
        mod $modname {
            use super::*;

            check!(i32x4, thermite::simd::i32x4<$backend>, i32, 4);
            check!(i32x8, thermite::simd::i32x8<$backend>, i32, 8);
            check!(u32x4, thermite::simd::u32x4<$backend>, u32, 4);
            check!(u32x8, thermite::simd::u32x8<$backend>, u32, 8);
            check!(u16x8, thermite::simd::u16x8<$backend>, u16, 8);
            check!(i16x16, thermite::simd::i16x16<$backend>, i16, 16);
            check!(u64x2, thermite::simd::u64x2<$backend>, u64, 2);
            check!(i64x4, thermite::simd::i64x4<$backend>, i64, 4);
            check!(u8x16, thermite::simd::u8x16<$backend>, u8, 16);
            // Floats: exact small integers, so the tree reassociation is exact too.
            check!(f32x4, thermite::simd::f32x4<$backend>, f32, 4);
            check!(f32x8, thermite::simd::f32x8<$backend>, f32, 8);
            check!(f64x2, thermite::simd::f64x2<$backend>, f64, 2);
            check!(f64x4, thermite::simd::f64x4<$backend>, f64, 4);

            /// The reason the ladder fills with a broadcast edge lane rather than
            /// `NumericRegister::MAX`/`MIN`: those are `f32::MAX`/`f32::MIN`, not
            /// `+/-inf`, so an infinite lane would be clamped to the finite bound.
            #[test]
            fn infinities() {
                type V = thermite::simd::f32x4<$backend>;

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

                // Same, from the other end.
                let r = V::new([3.0, 2.0, 1.0, f32::INFINITY]);
                assert_eq!(
                    r.reverse_prefix_min().into_array().as_slice(),
                    [1.0, 1.0, 1.0, f32::INFINITY].as_slice(),
                    "reverse_prefix_min must not clamp +inf to f32::MAX"
                );
            }

            /// A scan of one lane is the identity, and a scan of a constant vector is
            /// that constant (for min/max), cheap invariants that catch an off-by-one
            /// in the ladder's stage count.
            #[test]
            fn degenerate() {
                type V = thermite::simd::i32x8<$backend>;

                let c = V::splat(7);
                assert_eq!(c.prefix_min().into_array(), [7i32; 8].into());
                assert_eq!(c.prefix_max().into_array(), [7i32; 8].into());
                assert_eq!(c.reverse_prefix_min().into_array(), [7i32; 8].into());
                assert_eq!(c.reverse_prefix_max().into_array(), [7i32; 8].into());

                // 1,1,1.. summed forward is 1,2,3,.. and backward is 8,7,6,..
                let ones = V::splat(1);
                assert_eq!(ones.prefix_sum().into_array(), [1i32, 2, 3, 4, 5, 6, 7, 8].into());
                assert_eq!(
                    ones.reverse_prefix_sum().into_array(),
                    [8i32, 7, 6, 5, 4, 3, 2, 1].into()
                );
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

// The whole-array composition (carrying each chunk's total into the next) lives in
// `examples/prefix_sum_array.rs`, which self-checks against a sequential oracle over a
// spread of awkward lengths and start offsets:
//
//     cargo run --release --example prefix_sum_array
