//! `IntegerVector` bit-count exposure: `count_ones`/`count_zeros` and the
//! `leading_*`/`trailing_*` pairs.
//!
//! The register layer is already differentially tested against the scalar
//! oracle (`diff_polyfill`'s `for_lztz`), so what this suite guards is the
//! *vector-layer wiring*: each `Vector<R>` method must forward to the matching
//! register method. The realistic bug is a swap - `trailing_zeros` delegating
//! to `leading_zeros`, or `*_ones` to `*_zeros` - which no register-layer test
//! can see. Values are chosen so every one of the six answers differs
//! (asymmetric bit patterns, not palindromes), making a crossed wire a failure
//! rather than a coincidence.
//!
//! Uses the always-available scalar backend, plus the runtime-dispatched native
//! backend so the host's real registers are covered too.

use thermite::backend::scalar::Scalar;
use thermite::backend::scalar::prelude::*;

/// Bit patterns whose six counts are pairwise distinguishing: none is a bit
/// reversal of itself, and each mixes leading/trailing runs of both bits.
macro_rules! probes {
    ($elem:ty) => {{
        let max = <$elem>::MAX;
        let bits = <$elem>::BITS as $elem;
        [
            0,               // all-zero: the leading/trailing degenerate case
            max,             // all-one: the other degenerate case
            1,               // tz = 0, lz = BITS - 1
            1 << (bits - 1), // the mirror image of the above
            0b1011_0000,     // tz = 4, to = 0, lz depends on width
            0b0000_1101,     // tz = 0, to = 1
            max ^ 0b0111,    // trailing ones run, leading ones run
            max >> 1,        // lz = 1, tz = 0, to = BITS - 1
        ]
    }};
}

macro_rules! check {
    ($name:ident, $width:ident, $elem:ty) => {
        #[test]
        fn $name() {
            type V = thermite::simd::$width<Scalar>;
            const LANES: usize = <V as GenericVector>::LANES;

            let probes = probes!($elem);

            let mut i = 0;
            while i < probes.len() {
                let mut data = [<$elem>::default(); LANES];
                for lane in 0..LANES {
                    data[lane] = probes[(i + lane) % probes.len()];
                }
                let x = V::new(data.into());

                let got_co = x.count_ones();
                let got_cz = x.count_zeros();
                let got_lo = x.leading_ones();
                let got_lz = x.leading_zeros();
                let got_to = x.trailing_ones();
                let got_tz = x.trailing_zeros();

                for lane in 0..LANES {
                    let v = data[lane];
                    let ctx = format!("{} lane={lane} value={v:#x}", stringify!($width));

                    assert_eq!(
                        got_co.as_slice()[lane],
                        v.count_ones() as $elem,
                        "count_ones {ctx}"
                    );
                    assert_eq!(
                        got_cz.as_slice()[lane],
                        v.count_zeros() as $elem,
                        "count_zeros {ctx}"
                    );
                    assert_eq!(
                        got_lo.as_slice()[lane],
                        v.leading_ones() as $elem,
                        "leading_ones {ctx}"
                    );
                    assert_eq!(
                        got_lz.as_slice()[lane],
                        v.leading_zeros() as $elem,
                        "leading_zeros {ctx}"
                    );
                    assert_eq!(
                        got_to.as_slice()[lane],
                        v.trailing_ones() as $elem,
                        "trailing_ones {ctx}"
                    );
                    assert_eq!(
                        got_tz.as_slice()[lane],
                        v.trailing_zeros() as $elem,
                        "trailing_zeros {ctx}"
                    );
                }

                i += LANES;
            }
        }
    };
}

check!(bit_counts_u8x16, u8x16, u8);
check!(bit_counts_u16x8, u16x8, u16);
check!(bit_counts_u32x4, u32x4, u32);
check!(bit_counts_u32x8, u32x8, u32);
check!(bit_counts_u64x2, u64x2, u64);

/// The same wiring check on whatever backend the host actually dispatches to,
/// at the native width. Catches a per-backend register impl that the scalar
/// path would never touch.
#[test]
fn bit_counts_native() {
    let probes = probes!(u32);

    for &v in &probes {
        let [co, cz, lo, lz, to, tz] = thermite::dispatch_dyn!(for<S> |v: u32| -> [u32; 6] {
            let x = u32xN::splat(v);
            [
                x.count_ones().extract::<0>(),
                x.count_zeros().extract::<0>(),
                x.leading_ones().extract::<0>(),
                x.leading_zeros().extract::<0>(),
                x.trailing_ones().extract::<0>(),
                x.trailing_zeros().extract::<0>(),
            ]
        });

        assert_eq!(co, v.count_ones(), "count_ones value={v:#x}");
        assert_eq!(cz, v.count_zeros(), "count_zeros value={v:#x}");
        assert_eq!(lo, v.leading_ones(), "leading_ones value={v:#x}");
        assert_eq!(lz, v.leading_zeros(), "leading_zeros value={v:#x}");
        assert_eq!(to, v.trailing_ones(), "trailing_ones value={v:#x}");
        assert_eq!(tz, v.trailing_zeros(), "trailing_zeros value={v:#x}");
    }
}
