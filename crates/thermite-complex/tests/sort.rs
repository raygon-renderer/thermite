//! Lane sorts on `Complex<V>`: keyed on the lexicographic (re, im) order -
//! the same order `cmp_lt`/`min`/`max` already use.
//!
//! The `re` pattern contains duplicates ON PURPOSE, with distinct `im` values:
//! lexicographic order must break those ties by `im`, which is observable in
//! the output (unlike the key-with-payload composites, where tie order is
//! unspecified). The multiset check on whole (re, im) pairs still guards the
//! duplicated/dropped-pair failure mode independently.

use thermite::prelude::*;
use thermite::sort::{Ascending, Descending};
use thermite_complex::Complex;

fn re_part(lane: usize) -> f32 {
    const PATTERN: [f32; 16] = [5.0, 2.0, 9.0, 2.0, 7.0, 1.0, 8.0, 3.0, 6.0, 4.0, 0.0, 9.0, 1.0, 7.0, 3.0, 8.0];
    PATTERN[lane % 16]
}

/// Distinct per lane, and deliberately DESCENDING in the lane index so that a
/// sort which ignored `im` on tied `re` would leave tied groups reversed.
fn im_part(lane: usize) -> f32 {
    (64 - lane) as f32
}

fn encode(c: &Complex<f32>) -> (u32, u32) {
    (c.re.to_bits(), c.im.to_bits())
}

macro_rules! check {
    ($label:expr, $v:ty) => {{
        type C = Complex<$v>;

        let lanes = <C as GenericVector>::LANES;

        let mut v = C::default();
        let mut input: Vec<Complex<f32>> = Vec::with_capacity(lanes);
        for lane in 0..lanes {
            let e = Complex { re: re_part(lane), im: im_part(lane) };
            v = v.insertv(lane, e);
            input.push(e);
        }

        let got = |x: C| -> Vec<Complex<f32>> { (0..lanes).map(|i| x.extractv(i)).collect() };

        let verify = |label: &str, out: Vec<Complex<f32>>, ascending: bool| {
            // Lexicographic oracle: the EXACT output is determined, ties included.
            let mut want = input.clone();
            want.sort_by(|a, b| (a.re, a.im).partial_cmp(&(b.re, b.im)).unwrap());
            if !ascending {
                want.reverse();
            }
            let want: Vec<_> = want.iter().map(encode).collect();
            let got_e: Vec<_> = out.iter().map(encode).collect();
            assert_eq!(got_e, want, "{label}: not in lexicographic (re, im) order");
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
            sorted.sort_by(|a, b| (a.re, a.im).partial_cmp(&(b.re, b.im)).unwrap());
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
    check!("scalar Complex<f32>", Vector<f32>);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;

    #[test]
    fn v3() {
        use thermite::backend::x86_v3::{f32x4, f32x8, f32x16};

        check!("x86_v3 Complex<f32x4>", f32x4);
        check!("x86_v3 Complex<f32x8>", f32x8);
        // ArrayRegister chunks on AVX2.
        check!("x86_v3 Complex<f32x16>", f32x16);
    }

    #[test]
    fn v2() {
        use thermite::backend::x86_v2::f32x4;

        check!("x86_v2 Complex<f32x4>", f32x4);
    }
}
