//! Lane sorts on `Compensated<V>`: keyed on the lexicographic (value, error)
//! order - the same order `cmp_lt`/`min`/`max` already use.
//!
//! Values deliberately contain duplicates with distinct error terms, so the
//! lexicographic tie-break is observable in the output (the oracle is exact),
//! and a compare-exchange that duplicated or dropped one side of a tied pair
//! still fails the whole-pair multiset check independently.

use thermite::prelude::*;
use thermite::sort::{Ascending, Descending};
use thermite_compensated::Compensated;

fn value(lane: usize) -> f64 {
    const PATTERN: [f64; 16] = [
        5.0, 2.0, 9.0, 2.0, 7.0, 1.0, 8.0, 3.0, 6.0, 4.0, 0.0, 9.0, 1.0, 7.0, 3.0, 8.0,
    ];
    PATTERN[lane % 16]
}

/// Tiny and unique per lane: representable alongside any pattern value
/// without renormalization, and it tags which lane an error term came from.
fn error(lane: usize) -> f64 {
    (lane + 1) as f64 * 1e-12
}

fn encode(c: &Compensated<f64>) -> (u64, u64) {
    (c.value.to_bits(), c.error.to_bits())
}

macro_rules! check {
    ($label:expr, $v:ty) => {{
        type C = Compensated<$v>;

        let lanes = <C as GenericVector>::LANES;

        let mut v = C::default();
        let mut input: Vec<Compensated<f64>> = Vec::with_capacity(lanes);
        for lane in 0..lanes {
            let e = Compensated {
                value: value(lane),
                error: error(lane),
            };
            v = v.insertv(lane, e);
            input.push(e);
        }

        let got = |x: C| -> Vec<Compensated<f64>> { (0..lanes).map(|i| x.extractv(i)).collect() };

        let verify = |label: &str, out: Vec<Compensated<f64>>, ascending: bool| {
            // Lexicographic oracle: the EXACT output is determined, ties included.
            let mut want = input.clone();
            want.sort_by(|a, b| (a.value, a.error).partial_cmp(&(b.value, b.error)).unwrap());
            if !ascending {
                want.reverse();
            }
            let want: Vec<_> = want.iter().map(encode).collect();
            let got_e: Vec<_> = out.iter().map(encode).collect();
            assert_eq!(got_e, want, "{label}: not in lexicographic (value, error) order");
        };

        verify(
            &format!("{} asc", $label),
            got(<C as NumericVector>::sort_by::<Ascending>(v)),
            true,
        );
        verify(
            &format!("{} desc", $label),
            got(<C as NumericVector>::sort_by::<Descending>(v)),
            false,
        );

        if lanes >= 2 {
            let mut sorted = input.clone();
            sorted.sort_by(|a, b| (a.value, a.error).partial_cmp(&(b.value, b.error)).unwrap());
            let half = lanes / 2;
            let mut bitonic = sorted[..half].to_vec();
            let mut down = sorted[half..].to_vec();
            down.reverse();
            bitonic.extend(down);

            let mut bv = C::default();
            for (i, e) in bitonic.iter().enumerate() {
                bv = bv.insertv(i, *e);
            }
            verify(
                &format!("{} bitonic_clean", $label),
                got(<C as NumericVector>::bitonic_clean_by::<Ascending>(bv)),
                true,
            );
        }
    }};
}

#[test]
fn scalar() {
    check!("scalar Compensated<f64>", Vector<f64>);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;

    #[test]
    fn v3() {
        use thermite::backend::x86_v3::{f64x4, f64x8};

        check!("x86_v3 Compensated<f64x4>", f64x4);
        // ArrayRegister chunks on AVX2.
        check!("x86_v3 Compensated<f64x8>", f64x8);
    }

    #[test]
    fn v2() {
        use thermite::backend::x86_v2::f64x2;

        check!("x86_v2 Compensated<f64x2>", f64x2);
    }
}
