//! `UnsignedIntegerVector::in_range` correctness.
//!
//! `x.in_range(lo, hi)` is an inclusive unsigned range test `lo <= x <= hi`
//! (assuming `lo <= hi`), computed branchlessly as `(x - lo) <= (hi - lo)` with
//! *wrapping* subtraction. The interesting case is `x < lo`: the subtraction
//! must wrap to a large value so the lane tests `false` (a saturating sub would
//! wrongly test `true`). Verified against a scalar oracle, including the
//! boundaries and the `0`/`MAX` corners. Uses the always-available scalar backend.

use thermite::backend::scalar::Scalar;
use thermite::backend::scalar::prelude::*;

macro_rules! check {
    ($name:ident, $width:ident, $elem:ty) => {
        #[test]
        fn $name() {
            type V = thermite::simd::$width<Scalar>;
            const LANES: usize = <V as GenericVector>::LANES;

            // A spread of inclusive ranges, including degenerate (lo == hi),
            // a full range (0..=MAX), and ranges hugging both ends.
            let max = <$elem>::MAX;
            let ranges: [($elem, $elem); 7] =
                [(0, 0), (10, 20), (1, max), (0, max), (max, max), (max - 5, max), (64, 64)];

            // Lane values exercising below/at/inside/above each range plus the
            // wrap-prone 0 and MAX endpoints.
            let probes: [$elem; 12] =
                [0, 1, 9, 10, 15, 20, 21, 63, 64, 65, max - 1, max];

            for &(lo, hi) in &ranges {
                let lo_v = V::splat(lo);
                let hi_v = V::splat(hi);

                // Pack the probes into vectors LANES at a time.
                let mut i = 0;
                while i < probes.len() {
                    let mut data = [<$elem>::default(); LANES];
                    for lane in 0..LANES {
                        data[lane] = probes[(i + lane) % probes.len()];
                    }
                    let x = V::new(data.into());

                    // Mask -> {1, 0} per lane so we can diff via `as_slice`.
                    let got = x.in_range(lo_v, hi_v).select(V::splat(1), V::ZERO);

                    let mut want = [<$elem>::default(); LANES];
                    for lane in 0..LANES {
                        want[lane] = (data[lane] >= lo && data[lane] <= hi) as $elem;
                    }

                    for lane in 0..LANES {
                        assert_eq!(
                            got.as_slice()[lane],
                            want[lane],
                            "in_range lo={lo} hi={hi} x={} lane={lane}",
                            data[lane]
                        );
                    }

                    i += LANES;
                }
            }
        }
    };
}

check!(in_range_u8x16, u8x16, u8);
check!(in_range_u8x8, u8x8, u8);
check!(in_range_u16x8, u16x8, u16);
check!(in_range_u16x16, u16x16, u16);
check!(in_range_u32x4, u32x4, u32);
check!(in_range_u32x8, u32x8, u32);
check!(in_range_u64x2, u64x2, u64);
