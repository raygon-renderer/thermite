//! Lane sorts on `Dual<V, N>`: keyed on the primal, derivatives ride along.
//!
//! The key correctness hazard is a *tie by key*: a compare-exchange that
//! derives one routing mask and negates it for the other side sends the same
//! dual to both lanes of a tied pair - one composite duplicated, its partner
//! dropped. The primal pattern deliberately contains duplicates, and the
//! derivatives are distinct per lane, so that failure shows as a multiset
//! mismatch of whole duals even though the primal sequence still looks
//! perfectly sorted.

use thermite::prelude::*;
use thermite::sort::{Ascending, Descending};
use thermite_dual::Dual;

/// Duplicates on purpose - ties are the hazard.
fn primal(lane: usize) -> f32 {
    const PATTERN: [f32; 16] = [5.0, 2.0, 9.0, 2.0, 7.0, 1.0, 8.0, 3.0, 6.0, 4.0, 0.0, 9.0, 1.0, 7.0, 3.0, 8.0];
    PATTERN[lane % 16]
}

/// Unique per (lane, component), so every dual is distinguishable.
fn deriv(lane: usize, k: usize) -> f32 {
    (100 * (k + 1) + lane) as f32
}

/// Canonical encoding of a whole dual for multiset comparison.
fn encode<const N: usize>(d: &Dual<f32, N>) -> Vec<u32> {
    let mut v = vec![d.re.to_bits()];
    v.extend(d.dual.iter().map(|x| x.to_bits()));
    v
}

macro_rules! check {
    ($label:expr, $v:ty, $n:literal) => {{
        type D = Dual<$v, $n>;
        const N: usize = $n;

        let lanes = <D as GenericVector>::LANES;

        let mut v = D::default();
        let mut input: Vec<Dual<f32, N>> = Vec::with_capacity(lanes);
        for lane in 0..lanes {
            let e = Dual {
                re: primal(lane),
                dual: core::array::from_fn(|k| deriv(lane, k)),
            };
            v = v.insertv(lane, e);
            input.push(e);
        }

        let got = |x: D| -> Vec<Dual<f32, N>> { (0..lanes).map(|i| x.extractv(i)).collect() };

        let verify = |label: &str, out: Vec<Dual<f32, N>>, ascending: bool| {
            // 1. Primals are in order.
            let mut want_re: Vec<f32> = input.iter().map(|d| d.re).collect();
            want_re.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if !ascending {
                want_re.reverse();
            }
            let got_re: Vec<f32> = out.iter().map(|d| d.re).collect();
            assert_eq!(got_re, want_re, "{label}: primals not sorted");

            // 2. Every whole dual survived: multiset equality on full encodings,
            //    so a derivative paired with the wrong primal - or a tied pair
            //    duplicated/dropped - fails here.
            let mut a: Vec<_> = input.iter().map(encode).collect();
            let mut b: Vec<_> = out.iter().map(encode).collect();
            a.sort();
            b.sort();
            assert_eq!(a, b, "{label}: duals not a permutation of the input");
        };

        verify(
            &format!("{} asc", $label),
            got(<D as NumericVector>::sort_by::<Ascending>(v)),
            true,
        );
        verify(
            &format!("{} desc", $label),
            got(<D as NumericVector>::sort_by::<Descending>(v)),
            false,
        );

        // bitonic_clean on genuinely bitonic input: ascending run then
        // descending run of whole duals.
        if lanes >= 2 {
            let mut sorted = input.clone();
            sorted.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());
            let half = lanes / 2;
            let mut bitonic = sorted[..half].to_vec();
            let mut down = sorted[half..].to_vec();
            down.reverse();
            bitonic.extend(down);

            let mut bv = D::default();
            for (i, e) in bitonic.iter().enumerate() {
                bv = bv.insertv(i, *e);
            }
            verify(
                &format!("{} bitonic_clean", $label),
                got(<D as NumericVector>::bitonic_clean_by::<Ascending>(bv)),
                true,
            );
        }
    }};
}

#[test]
fn scalar() {
    check!("scalar Dual<f32, 1>", Vector<f32>, 1);
    check!("scalar Dual<f32, 2>", Vector<f32>, 2);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;

    #[test]
    fn v3() {
        use thermite::backend::x86_v3::{f32x4, f32x8, f32x16};

        check!("x86_v3 Dual<f32x4, 1>", f32x4, 1);
        check!("x86_v3 Dual<f32x8, 1>", f32x8, 1);
        check!("x86_v3 Dual<f32x8, 3>", f32x8, 3);
        // ArrayRegister chunks: the network's cross-chunk permutes on AVX2.
        check!("x86_v3 Dual<f32x16, 2>", f32x16, 2);
    }

    #[test]
    fn v2() {
        use thermite::backend::x86_v2::{f32x4, f32x8};

        check!("x86_v2 Dual<f32x4, 1>", f32x4, 1);
        check!("x86_v2 Dual<f32x8, 2>", f32x8, 2);
    }
}
