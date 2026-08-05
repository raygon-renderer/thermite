//! Prefix scans and the compress/expand family on `Complex<V>`.
//!
//! `prefix_sum` IS componentwise, because complex addition is. `prefix_min`/`max`
//! are not: the ordering is lexicographic over both parts (real first, then
//! imaginary), so a per-component scan would pair a real part from one lane with
//! an imaginary part from another. The pattern below fixes several duplicate real
//! parts with differing imaginary parts, which is exactly where that shows up.
//!
//! The compress/expand family is pure lane movement under a shared mask, so both
//! parts must take the same permutation.

use num_complex::Complex64;
use thermite::prelude::*;
use thermite_complex::Complex;

macro_rules! check {
    ($label:expr, $v:ty) => {{
        type C = Complex<$v>;

        let lanes = <C as GenericVector>::LANES;

        // Real parts repeat so the lexicographic tie-break on the imaginary part
        // is exercised; both are small exact integers so comparisons are exact.
        const RE: [f64; 16] = [
            5.0, 2.0, 9.0, 2.0, 7.0, 1.0, 8.0, 3.0, 6.0, 4.0, 0.0, 9.0, 1.0, 7.0, 3.0, 8.0,
        ];
        const IM: [f64; 16] = [
            3.0, 8.0, 1.0, 4.0, 9.0, 2.0, 7.0, 5.0, 0.0, 6.0, 2.0, 8.0, 5.0, 1.0, 9.0, 4.0,
        ];

        let mut v = C::default();
        let mut want: Vec<Complex64> = Vec::with_capacity(lanes);
        for lane in 0..lanes {
            let (re, im) = (RE[lane % 16], IM[lane % 16]);
            v = v.insertv(lane, Complex::new(re, im));
            want.push(Complex64::new(re, im));
        }

        let got = |x: C| -> Vec<Complex64> {
            (0..lanes)
                .map(|i| {
                    let e = x.extractv(i);
                    Complex64::new(e.re, e.im)
                })
                .collect()
        };

        // Lexicographic order, matching the vector impl's `cmp_lt`/`cmp_gt`.
        let lt = |a: Complex64, b: Complex64| (a.re, a.im) < (b.re, b.im);

        let mut acc = Complex64::new(0.0, 0.0);
        let fwd_sum: Vec<_> = want
            .iter()
            .map(|&e| {
                acc += e;
                acc
            })
            .collect();
        assert_eq!(got(v.prefix_sum()), fwd_sum, "{}: prefix_sum", $label);

        let mut acc = Complex64::new(0.0, 0.0);
        let mut rev_sum = vec![acc; lanes];
        for i in (0..lanes).rev() {
            acc += want[i];
            rev_sum[i] = acc;
        }
        assert_eq!(got(v.reverse_prefix_sum()), rev_sum, "{}: reverse_prefix_sum", $label);

        let mut acc = want[0];
        let fwd_min: Vec<_> = want
            .iter()
            .enumerate()
            .map(|(i, &e)| {
                if i > 0 && lt(e, acc) {
                    acc = e;
                }
                acc
            })
            .collect();
        assert_eq!(got(v.prefix_min()), fwd_min, "{}: prefix_min", $label);

        let mut acc = want[0];
        let fwd_max: Vec<_> = want
            .iter()
            .enumerate()
            .map(|(i, &e)| {
                if i > 0 && lt(acc, e) {
                    acc = e;
                }
                acc
            })
            .collect();
        assert_eq!(got(v.prefix_max()), fwd_max, "{}: prefix_max", $label);

        let mut acc = want[lanes - 1];
        let mut rev_min = vec![acc; lanes];
        for i in (0..lanes - 1).rev() {
            if lt(want[i], acc) {
                acc = want[i];
            }
            rev_min[i] = acc;
        }
        assert_eq!(got(v.reverse_prefix_min()), rev_min, "{}: reverse_prefix_min", $label);

        let mut acc = want[lanes - 1];
        let mut rev_max = vec![acc; lanes];
        for i in (0..lanes - 1).rev() {
            if lt(acc, want[i]) {
                acc = want[i];
            }
            rev_max[i] = acc;
        }
        assert_eq!(got(v.reverse_prefix_max()), rev_max, "{}: reverse_prefix_max", $label);

        // --- compress / expand ------------------------------------------------
        let mut src = C::default();
        for lane in 0..lanes {
            src = src.insertv(lane, Complex::new(-1.0 - lane as f64, -100.0 - lane as f64));
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
                sel = sel.insertv(lane, Complex::new(if on { 1.0 } else { 0.0 }, 0.0));
            }
            let mask = sel.cmp_gt(C::ZERO);

            assert_eq!(
                got(v.compress(mask).expand(mask)),
                want,
                "{}: expand(compress) bits={bits:b}",
                $label
            );
            assert_eq!(
                got(v.expand(mask).compress(mask)),
                want,
                "{}: compress(expand) bits={bits:b}",
                $label
            );
            assert_eq!(
                got(v.expand_z(mask)),
                got(mask.select(v.expand(mask), C::ZERO)),
                "{}: expand_z bits={bits:b}",
                $label
            );
            assert_eq!(
                got(v.expand_m(src, mask)),
                got(mask.select(v.expand(mask), src)),
                "{}: expand_m bits={bits:b}",
                $label
            );

            let count = (0..lanes).filter(|&l| (bits >> (l % 64)) & 1 == 1).count();
            let selected: Vec<_> = (0..lanes)
                .filter(|&l| (bits >> (l % 64)) & 1 == 1)
                .map(|l| want[l])
                .collect();
            let src_arr = got(src);
            let cm: Vec<_> = (0..lanes)
                .map(|i| if i < count { selected[i] } else { src_arr[i] })
                .collect();
            assert_eq!(got(v.compress_m(src, mask)), cm, "{}: compress_m bits={bits:b}", $label);

            assert_eq!(
                got(v.compress_z(mask).expand_m(src, mask)),
                got(mask.select(v, src)),
                "{}: compress_z + expand_m bits={bits:b}",
                $label
            );
        }
    }};
}

/// `Complex` must report its inner vector's align capability, in both directions:
/// the min/max scans are an `align` ladder, and a wrong answer here picks the wrong
/// lowering without changing any result.
#[test]
fn forwards_native_align() {
    assert!(
        !<Complex<Vector<f64>> as GenericVector>::HAS_NATIVE_ALIGN,
        "scalar backend has no native align; Complex must not claim one"
    );

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use thermite::backend::x86_v3::f64x4;
        assert!(
            <Complex<f64x4> as GenericVector>::HAS_NATIVE_ALIGN,
            "f64x4 has a native align on AVX2; Complex dropped it"
        );
    }
}

#[test]
fn scalar() {
    check!("scalar Complex<f64>", Vector<f64>);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;

    #[test]
    fn v3() {
        use thermite::backend::x86_v3::{f64x2, f64x4, f64x8};

        check!("x86_v3 Complex<f64x2>", f64x2);
        check!("x86_v3 Complex<f64x4>", f64x4);
        check!("x86_v3 Complex<f64x8>", f64x8);
    }
}
