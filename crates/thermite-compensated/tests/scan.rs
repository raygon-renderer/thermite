//! Prefix scans and the compress/expand family on `Compensated<V>`.
//!
//! The scans are the interesting half. Scanning `value` and `error` with the
//! inner vector's own scan would add the value lanes without ever renormalising
//! the carried error into them - plausible-looking floats, no compensation - so
//! `prefix_sum` runs the ladder on the double-double `+` instead.
//!
//! The test that pins this down is `catastrophic_cancellation`: a running sum
//! whose partials cancel down to a value a plain `f64` scan cannot represent.
//! Componentwise scanning passes every other check here and fails that one.
//!
//! `prefix_min`/`max` order by the compensated value and carry the winning
//! lane's error with it, so they too scan whole elements.

use thermite::prelude::*;
use thermite_compensated::Compensated;

macro_rules! check {
    ($label:expr, $v:ty) => {{
        type C = Compensated<$v>;

        let lanes = <C as GenericVector>::LANES;

        // Small exact integers, non-monotonic so a dropped ladder stage shows up.
        const PATTERN: [f64; 16] = [5.0, 2.0, 9.0, 2.0, 7.0, 1.0, 8.0, 3.0, 6.0, 4.0, 0.0, 9.0, 1.0, 7.0, 3.0, 8.0];

        let mut v = C::default();
        let mut want = Vec::with_capacity(lanes);
        for lane in 0..lanes {
            let e = PATTERN[lane % 16];
            v = v.insertv(lane, Compensated::new(e));
            want.push(e);
        }

        // Compare through the compensated value; the error term is checked by the
        // cancellation test below, where it is the only thing that can carry the answer.
        let got = |x: C| -> Vec<f64> { (0..lanes).map(|i| x.extractv(i).value()).collect() };

        let mut acc = 0.0;
        let fwd_sum: Vec<f64> = want
            .iter()
            .map(|&e| {
                acc += e;
                acc
            })
            .collect();
        assert_eq!(got(v.prefix_sum()), fwd_sum, "{}: prefix_sum", $label);

        let mut acc = 0.0;
        let mut rev_sum = vec![0.0; lanes];
        for i in (0..lanes).rev() {
            acc += want[i];
            rev_sum[i] = acc;
        }
        assert_eq!(got(v.reverse_prefix_sum()), rev_sum, "{}: reverse_prefix_sum", $label);

        let mut acc = f64::INFINITY;
        let fwd_min: Vec<f64> = want
            .iter()
            .map(|&e| {
                acc = acc.min(e);
                acc
            })
            .collect();
        assert_eq!(got(v.prefix_min()), fwd_min, "{}: prefix_min", $label);

        let mut acc = f64::NEG_INFINITY;
        let fwd_max: Vec<f64> = want
            .iter()
            .map(|&e| {
                acc = acc.max(e);
                acc
            })
            .collect();
        assert_eq!(got(v.prefix_max()), fwd_max, "{}: prefix_max", $label);

        let mut acc = f64::INFINITY;
        let mut rev_min = vec![0.0; lanes];
        for i in (0..lanes).rev() {
            acc = acc.min(want[i]);
            rev_min[i] = acc;
        }
        assert_eq!(got(v.reverse_prefix_min()), rev_min, "{}: reverse_prefix_min", $label);

        let mut acc = f64::NEG_INFINITY;
        let mut rev_max = vec![0.0; lanes];
        for i in (0..lanes).rev() {
            acc = acc.max(want[i]);
            rev_max[i] = acc;
        }
        assert_eq!(got(v.reverse_prefix_max()), rev_max, "{}: reverse_prefix_max", $label);

        // --- compress / expand: pure lane movement, both components together ---
        let mut src = C::default();
        for lane in 0..lanes {
            src = src.insertv(lane, Compensated::new(-1.0 - lane as f64));
        }

        let patterns: Vec<u64> = if lanes <= 8 {
            (0..(1u64 << lanes)).collect()
        } else {
            let mut s = 0x9E3779B97F4A7C15u64;
            (0..256)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    s
                })
                .chain([0, u64::MAX])
                .collect()
        };

        for bits in patterns {
            let mut sel = C::default();
            for lane in 0..lanes {
                let on = (bits >> (lane % 64)) & 1 == 1;
                sel = sel.insertv(lane, Compensated::new(if on { 1.0 } else { 0.0 }));
            }
            let mask = sel.cmp_gt(C::ZERO);

            assert_eq!(got(v.compress(mask).expand(mask)), want, "{}: expand(compress) bits={bits:b}", $label);
            assert_eq!(got(v.expand(mask).compress(mask)), want, "{}: compress(expand) bits={bits:b}", $label);
            assert_eq!(
                got(v.expand_m(src, mask)),
                got(mask.select(v.expand(mask), src)),
                "{}: expand_m bits={bits:b}",
                $label
            );

            let count = (0..lanes).filter(|&l| (bits >> (l % 64)) & 1 == 1).count();
            let selected: Vec<f64> = (0..lanes).filter(|&l| (bits >> (l % 64)) & 1 == 1).map(|l| want[l]).collect();
            let src_arr = got(src);
            let cm: Vec<f64> = (0..lanes)
                .map(|i| if i < count { selected[i] } else { src_arr[i] })
                .collect();
            assert_eq!(got(v.compress_m(src, mask)), cm, "{}: compress_m bits={bits:b}", $label);
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
        use thermite::backend::x86_v3::{f64x2, f64x4, f64x8};

        check!("x86_v3 Compensated<f64x2>", f64x2);
        check!("x86_v3 Compensated<f64x4>", f64x4);
        check!("x86_v3 Compensated<f64x8>", f64x8);
    }
}

/// The scan has to be compensated, not componentwise.
///
/// `1 + eps + ... - 1` where each `eps` is far below the ulp of the running
/// partial: a plain `f64` scan rounds every `1 + eps` back to `1.0` and ends at
/// exactly `0.0`, while the double-double keeps the epsilons in the error term
/// and recovers their sum once the `1.0` cancels out.
#[test]
fn catastrophic_cancellation() {
    use thermite::backend::x86_v3::f64x8;
    type C = Compensated<f64x8>;

    const EPS: f64 = 1.0e-18;

    // [1, eps, eps, eps, eps, eps, eps, -1]
    let mut v = C::default();
    v = v.insertv(0, Compensated::new(1.0));
    for lane in 1..7 {
        v = v.insertv(lane, Compensated::new(EPS));
    }
    v = v.insertv(7, Compensated::new(-1.0));

    let last = v.prefix_sum().extractv(7);

    // Naive f64 accumulation gives exactly 0.0 here.
    let mut naive = 0.0f64;
    for lane in 0..8 {
        naive += v.extractv(lane).value();
    }
    assert_eq!(naive, 0.0, "the naive f64 scan is expected to lose the epsilons");

    // The compensated scan recovers 6 * EPS.
    let want = 6.0 * EPS;
    let got = last.value + last.error;
    assert!(
        (got - want).abs() <= want * 1e-12,
        "compensated prefix_sum lost the error term: got {got:e}, want {want:e}"
    );
}
