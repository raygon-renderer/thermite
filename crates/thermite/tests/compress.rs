//! `GenericVector::compress` / `compress_z` (left-pack) correctness.
//!
//! `compress(mask)` is a stable partition: lanes where `mask` is `true` packed
//! to the front in order, the unselected lanes kept in the tail in order.
//! `compress_z(mask)` is the same but zero-fills the tail (AVX-512 zero-masking
//! semantics). Verified against scalar oracles over every possible mask pattern
//! for small lane counts. Uses the always-available scalar backend.

mod harness;

use thermite::prelude::*;

macro_rules! check {
    ($($name:ident: $ty:ty, $lanes:literal),+ $(,)?) => {
        for_each_backend_concrete! {$(
        fn $name() {
            type V = $ty;
            const LANES: usize = $lanes;

            // Distinct, nonzero lane values so a misplaced/dropped lane is visible.
            let mut data = [<V as GenericVector>::Element::default(); LANES];
            for i in 0..LANES {
                data[i] = ((i + 1) * 10) as _;
            }
            let v = V::new(data.into());

            for bits in 0u32..(1 << LANES) {
                // Build the mask from the bit pattern: 1 where selected, then
                // `!= 0` lifts it into the mask domain.
                let mut sel = [<V as GenericVector>::Element::default(); LANES];
                for lane in 0..LANES {
                    if (bits >> lane) & 1 == 1 {
                        sel[lane] = 1 as _;
                    }
                }
                let m = V::new(sel.into()).cmp_ne(V::ZERO);

                // Non-zeroing oracle: selected lanes in order, then unselected
                // lanes in order.
                let mut partition = [<V as GenericVector>::Element::default(); LANES];
                let mut pos = 0;
                for lane in 0..LANES {
                    if (bits >> lane) & 1 == 1 {
                        partition[pos] = data[lane];
                        pos += 1;
                    }
                }
                let count = pos;
                for lane in 0..LANES {
                    if (bits >> lane) & 1 == 0 {
                        partition[pos] = data[lane];
                        pos += 1;
                    }
                }

                // Zeroing oracle: selected lanes in order, then zeros.
                let mut zeroed = [<V as GenericVector>::Element::default(); LANES];
                zeroed[..count].copy_from_slice(&partition[..count]);

                let got = v.compress(m);
                let got_z = v.compress_z(m);
                for lane in 0..LANES {
                    assert_eq!(
                        got.as_slice()[lane],
                        partition[lane],
                        "compress bits={bits:0width$b} lane={lane}",
                        width = LANES
                    );
                    assert_eq!(
                        got_z.as_slice()[lane],
                        zeroed[lane],
                        "compress_z bits={bits:0width$b} lane={lane}",
                        width = LANES
                    );
                }
            }
        }
        )+}
    };
}

check! {
    compress_f32x4: f32x4, 4,
    compress_i32x4: i32x4, 4,
    compress_u32x4: u32x4, 4,
    compress_f32x8: f32x8, 8,
    compress_f64x4: f64x4, 4,
    compress_i64x2: i64x2, 2,
    compress_i32x16: i32x16, 16,
    compress_i16x8: thermite::simd::i16x8<S>, 8,
    compress_u8x16: thermite::simd::u8x16<S>, 16,
}
