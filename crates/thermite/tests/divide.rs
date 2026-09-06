#![allow(clippy::unnecessary_cast)]
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use thermite::divider::{BranchfreeDivider, Divider};
use thermite::prelude::*;
use thermite::simd::{Simd, i32x8, i64x8, u32x8, u64x8};

/// Every divisor under test: the whole `i8` range plus the `i32` extremes.
fn divisors() -> impl Iterator<Item = i64> {
    ((i8::MIN as i64)..=(i8::MAX as i64)).chain([i32::MIN as i64, i32::MAX as i64])
}

/// Exhaustively check one divisor against the whole `i16` numerator range (plus
/// the `i32` extremes), for all four widths, scalar and vector, plain and
/// branchfree.
///
/// The dividers are built once per `d`, not once per `(d, i)`: they are pure
/// functions of `d`, and reconstructing them 65k times was most of the runtime.
#[inline(always)]
fn check_divisor<S: Simd>(d: i64) {
    if d == 0 {
        return;
    }

    let d_i32 = Divider::i32(d as i32);
    let d_i64 = Divider::i64(d as i64);
    let d_u32 = Divider::u32(d as u32);
    let d_u64 = Divider::u64(d as u64);

    let d_i32_bf = BranchfreeDivider::i32(d as i32);
    let d_i64_bf = BranchfreeDivider::i64(d as i64);

    // The unsigned branchfree divider does not support 1.
    let unsigned_bf = (d != 1).then(|| (BranchfreeDivider::u32(d as u32), BranchfreeDivider::u64(d as u64)));

    for i in ((i16::MIN as i64)..=(i16::MAX as i64)).chain([i32::MIN as i64, i32::MAX as i64]) {
        let x_i32 = i32x8::<S>::splat(i as i32) + i32x8::<S>::indexed();
        let x_i64 = i64x8::<S>::splat(i as i64) + i64x8::<S>::indexed();
        let x_u32 = u32x8::<S>::splat(i as u32) + u32x8::<S>::indexed();
        let x_u64 = u64x8::<S>::splat(i as u64) + u64x8::<S>::indexed();

        let expected_i32 = x_i32.map(|v| v.wrapping_div(d as i32));
        let expected_i64 = x_i64.map(|v| v.wrapping_div(d as i64));
        let expected_u32 = x_u32.map(|v| v.wrapping_div(d as u32));
        let expected_u64 = x_u64.map(|v| v.wrapping_div(d as u64));

        assert_eq!(d_i32.divide(i as i32), (i as i32).wrapping_div(d as i32));
        assert_eq!(d_i64.divide(i as i64), (i as i64).wrapping_div(d as i64));
        assert_eq!(d_u32.divide(i as u32), (i as u32).wrapping_div(d as u32));
        assert_eq!(d_u64.divide(i as u64), (i as u64).wrapping_div(d as u64));

        assert_eq!(d_i32_bf.divide(i as i32), (i as i32).wrapping_div(d as i32));
        assert_eq!(d_i64_bf.divide(i as i64), (i as i64).wrapping_div(d as i64));

        let result_i32_bf = x_i32 / d_i32_bf;
        let result_i64_bf = x_i64 / d_i64_bf;
        let result_i32 = x_i32 / d_i32;
        let result_i64 = x_i64 / d_i64;

        let result_u32 = x_u32 / d_u32;
        let result_u64 = x_u64 / d_u64;

        assert_eq!(expected_i32, result_i32, "i32 division failed for d={d}, x={x_i32:?}");
        assert_eq!(expected_i64, result_i64, "i64 division failed for d={d}, x={x_i64:?}");

        assert_eq!(
            expected_i32, result_i32_bf,
            "i32 bf division failed for d={d}, x={x_i32:?}"
        );
        assert_eq!(
            expected_i64, result_i64_bf,
            "i64 bf division failed for d={d}, x={x_i64:?}"
        );

        assert_eq!(
            expected_u32, result_u32,
            "u32 division failed for d={}, x={x_u32:?} ({d}, {x_i32:?})",
            d as u32
        );
        assert_eq!(
            expected_u64, result_u64,
            "u64 division failed for d={}, x={x_u64:?} ({d}, {x_i32:?})",
            d as u64
        );

        let Some((d_u32_bf, d_u64_bf)) = unsigned_bf else {
            continue;
        };

        assert_eq!(d_u32_bf.divide(i as u32), (i as u32).wrapping_div(d as u32));
        assert_eq!(d_u64_bf.divide(i as u64), (i as u64).wrapping_div(d as u64));

        let result_u32_bf = x_u32 / d_u32_bf;
        let result_u64_bf = x_u64 / d_u64_bf;

        assert_eq!(
            expected_u32, result_u32_bf,
            "u32 bf division failed for d={}, x={x_u32:?} ({d}, {x_i32:?})",
            d as u32
        );
        assert_eq!(
            expected_u64, result_u64_bf,
            "u64 bf division failed for d={}, x={x_u64:?} ({d}, {x_i32:?})",
            d as u64
        );
    }
}

/// The divisor sweep is ~17M exhaustive checks, by far the longest test in the
/// suite, and long enough that it alone set the wall clock of a fully parallel
/// run. Splitting it into interleaved shards (divisor `k`, `k + SHARDS`, ...)
/// lets the harness run them concurrently. The coverage is identical, and the
/// stride keeps each shard's mix of easy/hard divisors even.
const SHARDS: usize = 16;

#[inline(always)]
fn run_shard<S: Simd>(shard: usize) {
    if cfg!(debug_assertions) {
        println!("Skipping divider tests in debug mode, run in release mode for full coverage.");
        return;
    }

    for d in divisors().skip(shard).step_by(SHARDS) {
        check_divisor::<S>(d);
    }
}

macro_rules! divide_shards {
    ($($name:ident = $shard:expr),+ $(,)?) => {
        for_each_backend_concrete! {
            // Edge case that failed before it was fixed
            fn test_failure1() {
                let d = 4294967168u32;
                let n = u32x4::new([4294967165, 4294967166, 4294967167, 4294967168]);

                let expected = n.map(|v| v / d);

                let d_divider = BranchfreeDivider::u32(d);
                let result = n / d_divider;

                assert_eq!(expected, result, "u32 division failed for d={}, n={:?}", d, n);
            }

            $(
                fn $name() {
                    run_shard::<S>($shard);
                }
            )+
        }
    };
}

divide_shards! {
    test_divide_00 = 0, test_divide_01 = 1, test_divide_02 = 2, test_divide_03 = 3,
    test_divide_04 = 4, test_divide_05 = 5, test_divide_06 = 6, test_divide_07 = 7,
    test_divide_08 = 8, test_divide_09 = 9, test_divide_10 = 10, test_divide_11 = 11,
    test_divide_12 = 12, test_divide_13 = 13, test_divide_14 = 14, test_divide_15 = 15,
}
