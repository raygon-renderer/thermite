//! `GenericVector::prefix_mask` / `suffix_mask` correctness.
//!
//! These are the canonical tail-handling mask constructors: `prefix_mask(n)`
//! selects the first `n` lanes, `suffix_mask(n)` the last `n`, with `n`
//! clamped to `LANES`. Verified against a per-lane boolean oracle by selecting
//! `ONE` vs `ZERO` through the produced mask and reading lanes back. Uses the
//! always-available scalar backend so it runs on every target.

mod harness;

use thermite::prelude::*;

macro_rules! check {
    ($($name:ident: $ty:ty, $lanes:literal),+ $(,)?) => {
        for_each_backend_concrete! {$(
        fn $name() {
            type V = $ty;
            const LANES: usize = $lanes;

            for n in 0..=LANES {
                // prefix: lanes [0, n) true.
                let v = V::prefix_mask(n).select(V::ONE, V::ZERO);
                for lane in 0..LANES {
                    let expected = (lane < n) as i64;
                    assert_eq!(v.as_slice()[lane] as i64, expected, "prefix n={n} lane={lane}");
                }

                // suffix: lanes [LANES - n, LANES) true.
                let v = V::suffix_mask(n).select(V::ONE, V::ZERO);
                for lane in 0..LANES {
                    let expected = (lane >= LANES - n) as i64;
                    assert_eq!(v.as_slice()[lane] as i64, expected, "suffix n={n} lane={lane}");
                }
            }

            // n == 0 is empty, n == LANES (and beyond, clamped) is full.
            assert!(V::prefix_mask(0).none(), "prefix(0) should be empty");
            assert!(V::suffix_mask(0).none(), "suffix(0) should be empty");
            assert!(V::prefix_mask(LANES + 7).all(), "prefix clamps to all-true");
            assert!(V::suffix_mask(LANES + 7).all(), "suffix clamps to all-true");
        }
        )+}
    };
}

check! {
    prefix_suffix_f32x4: f32x4, 4,
    prefix_suffix_f32x8: f32x8, 8,
    prefix_suffix_u32x4: u32x4, 4,
    prefix_suffix_f64x4: f64x4, 4,
    prefix_suffix_i32x4: i32x4, 4,
    prefix_suffix_f32x16: f32x16, 16,
    prefix_suffix_i16x8: thermite::simd::i16x8<S>, 8,
    prefix_suffix_u8x16: thermite::simd::u8x16<S>, 16,
    prefix_suffix_i64x2: i64x2, 2,
}
