//! `CastRegister` / `CastVector` reference-oracle checks.
//!
//! Narrowing, same-signedness casts clamp out-of-range source values into the
//! destination element range (instead of `as`-style wrapping). Verified against
//! an inline clamp oracle. Uses the always-available scalar backend, exercising
//! both the 1-lane leaf impls and the `ArrayRegister` element-wise delegation.

mod harness;

use thermite::prelude::*;

/// Splat each input, saturating-cast, compare to the `clamp as` oracle.
macro_rules! leaf {
    ($fromv:ty => $tov:ty, $from:ty => $to:ty, [$($in:expr),+ $(,)?]) => {
        $(
            let input: $from = $in;
            let expected: $to = input.clamp(<$to>::MIN as $from, <$to>::MAX as $from) as $to;
            let got = <$fromv>::splat(input).saturating_cast::<$tov>().extract::<0>();
            assert_eq!(got, expected, "{} -> {}: input={input}", stringify!($fromv), stringify!($tov));
        )+
    };
}

for_each_backend_concrete! {
    // signed narrowing, adjacent + skip-level
    fn signed_narrowing() {
        leaf!(i64x4 => i32x4, i64 => i32, [0, 5, -3, i32::MAX as i64, i32::MIN as i64, i64::MAX, i64::MIN, 5_000_000_000]);
        leaf!(i32x8 => thermite::simd::i16x8<S>, i32 => i16, [0, 5, -3, 32767, -32768, 100_000, -100_000]);
        leaf!(thermite::simd::i16x8<S> => thermite::simd::i8x8<S>, i16 => i8, [0, 5, -3, 127, -128, 1000, -1000]);
        leaf!(i64x4 => thermite::simd::i8x4<S>, i64 => i8, [0, 42, -42, 127, -128, i64::MAX, i64::MIN]);
    }

    // unsigned narrowing, adjacent + skip-level (high-end clamp only)
    fn unsigned_narrowing() {
        leaf!(u64x4 => u32x4, u64 => u32, [0, 5, u32::MAX as u64, u64::MAX, 5_000_000_000]);
        leaf!(u32x8 => thermite::simd::u16x8<S>, u32 => u16, [0, 5, 65535, 100_000]);
        leaf!(thermite::simd::u16x8<S> => thermite::simd::u8x8<S>, u16 => u8, [0, 5, 255, 1000]);
        leaf!(u64x4 => thermite::simd::u8x4<S>, u64 => u8, [0, 42, 255, u64::MAX]);
    }

    // Multi-lane delegation through `ArrayRegister` / the native narrowing.
    fn multilane_i64x4_to_i32x4() {
        let v = i64x4::new([5_i64, i64::MAX, -i64::MAX, -3]);
        let r = v.saturating_cast::<i32x4>();
        assert_eq!(r.as_slice(), [5, i32::MAX, i32::MIN, -3].as_slice());
    }

    fn multilane_u64x4_to_u32x4() {
        let v = u64x4::new([7_u64, u64::MAX, 0, 100]);
        let r = v.saturating_cast::<u32x4>();
        assert_eq!(r.as_slice(), [7, u32::MAX, 0, 100].as_slice());
    }
}
