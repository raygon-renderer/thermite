#![allow(clippy::unnecessary_cast)]

use thermite::{
    backend::x86_v3::prelude::*,
    divider::{BranchfreeDivider, Divider},
};

// Edge case that failed before it was fixed
#[test]
fn test_failure1() {
    let d = 4294967168u32;
    let n = u32x4::new([4294967165, 4294967166, 4294967167, 4294967168]);

    let expected = n.map(|v| v / d);

    let d_divider = BranchfreeDivider::u32(d);
    let result = n / d_divider;

    assert_eq!(expected, result, "u32 division failed for d={}, n={:?}", d, n);
}

#[test]
fn test_divide() {
    if cfg!(debug_assertions) {
        println!("Skipping divider tests in debug mode, run in release mode for full coverage.");
        return;
    }

    for d in ((i8::MIN as i64)..=(i8::MAX as i64)).chain([i32::MIN as i64, i32::MAX as i64]) {
        for i in ((i16::MIN as i64)..=(i16::MAX as i64)).chain([i32::MIN as i64, i32::MAX as i64]) {
            if d == 0 {
                continue;
            }

            let x_i32 = i32x8::splat(i as i32) + i32x8::indexed();
            let x_i64 = i64x8::splat(i as i64) + i64x8::indexed();
            let x_u32 = u32x8::splat(i as u32) + u32x8::indexed();
            let x_u64 = u64x8::splat(i as u64) + u64x8::indexed();

            let expected_i32 = x_i32.map(|v| v.wrapping_div(d as i32));
            let expected_i64 = x_i64.map(|v| v.wrapping_div(d as i64));
            let expected_u32 = x_u32.map(|v| v.wrapping_div(d as u32));
            let expected_u64 = x_u64.map(|v| v.wrapping_div(d as u64));

            let d_i32 = Divider::i32(d as i32);
            let d_i64 = Divider::i64(d as i64);
            let d_u32 = Divider::u32(d as u32);
            let d_u64 = Divider::u64(d as u64);

            assert_eq!(d_i32.divide(i as i32), (i as i32).wrapping_div(d as i32));
            assert_eq!(d_i64.divide(i as i64), (i as i64).wrapping_div(d as i64));
            assert_eq!(d_u32.divide(i as u32), (i as u32).wrapping_div(d as u32));
            assert_eq!(d_u64.divide(i as u64), (i as u64).wrapping_div(d as u64));

            let d_i32_bf = BranchfreeDivider::i32(d as i32);
            let d_i64_bf = BranchfreeDivider::i64(d as i64);

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

            if d == 1 {
                continue; // unsigned branchfree divider for 1 is not supported
            }

            let d_u32_bf = BranchfreeDivider::u32(d as u32);
            let d_u64_bf = BranchfreeDivider::u64(d as u64);

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
}
