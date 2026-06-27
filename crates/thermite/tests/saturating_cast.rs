//! `SaturatingCastRegister` / `SaturatingCastVector` reference-oracle checks.
//!
//! Narrowing, same-signedness casts clamp out-of-range source values into the
//! destination element range (instead of `as`-style wrapping). Verified against
//! an inline clamp oracle. Uses the always-available scalar backend, exercising
//! both the 1-lane leaf impls and the `ArrayRegister` element-wise delegation.

use thermite::backend::scalar::prelude::*;

/// 1-lane leaf: splat each input, saturating-cast, compare to `clamp as` oracle.
macro_rules! leaf {
    ($name:ident, $from:ty => $to:ty, [$($in:expr),+ $(,)?]) => {
        #[test]
        fn $name() {
            $(
                let input: $from = $in;
                let expected: $to = input.clamp(<$to>::MIN as $from, <$to>::MAX as $from) as $to;
                let got = Vector::<$from>::splat(input).saturating_cast::<Vector<$to>>().extract::<0>();
                assert_eq!(got, expected, "{} -> {}: input={input}", stringify!($from), stringify!($to));
            )+
        }
    };
}

// signed narrowing, adjacent + skip-level
leaf!(i64_to_i32, i64 => i32, [0, 5, -3, i32::MAX as i64, i32::MIN as i64, i64::MAX, i64::MIN, 5_000_000_000]);
leaf!(i32_to_i16, i32 => i16, [0, 5, -3, 32767, -32768, 100_000, -100_000]);
leaf!(i16_to_i8,  i16 => i8,  [0, 5, -3, 127, -128, 1000, -1000]);
leaf!(i64_to_i8,  i64 => i8,  [0, 42, -42, 127, -128, i64::MAX, i64::MIN]);

// unsigned narrowing, adjacent + skip-level (high-end clamp only)
leaf!(u64_to_u32, u64 => u32, [0, 5, u32::MAX as u64, u64::MAX, 5_000_000_000]);
leaf!(u32_to_u16, u32 => u16, [0, 5, 65535, 100_000]);
leaf!(u16_to_u8,  u16 => u8,  [0, 5, 255, 1000]);
leaf!(u64_to_u8,  u64 => u8,  [0, 42, 255, u64::MAX]);

// Multi-lane delegation through `ArrayRegister`.
#[test]
fn multilane_i64x4_to_i32x4() {
    let v = i64x4::new([5_i64, i64::MAX, -i64::MAX, -3]);
    let r = v.saturating_cast::<i32x4>();
    assert_eq!(r.as_slice(), [5, i32::MAX, i32::MIN, -3].as_slice());
}

#[test]
fn multilane_u64x4_to_u32x4() {
    let v = u64x4::new([7_u64, u64::MAX, 0, 100]);
    let r = v.saturating_cast::<u32x4>();
    assert_eq!(r.as_slice(), [7, u32::MAX, 0, 100].as_slice());
}
