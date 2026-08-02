//! Prefix scans and the compress/expand family on `Dual<V, N>`.
//!
//! Both families are places where "delegate to each component" is right for one
//! operation and wrong for its neighbour, so each is checked against a scalar
//! oracle built from whole dual elements rather than from components:
//!
//! - `prefix_sum` IS componentwise (sum is linear), and the oracle confirms the
//!   derivative lanes accumulate alongside their primals.
//! - `prefix_min`/`max` are NOT. A dual is ordered by its primal and carries the
//!   winner's derivative, so a componentwise scan would pair a primal from one
//!   lane with a derivative from another. Values are chosen so the argmin lane
//!   differs from the arg-derivative lane, which is what makes that visible.
//! - the compress/expand family is pure lane movement under a shared mask, so it
//!   must apply the *same* permutation to every component.
//!
//! Widths sweep 1/4/8/16 lanes so the doubling ladder runs 0 through 4 stages,
//! including the `ArrayRegister` width on AVX2.

use thermite::prelude::*;
use thermite_dual::Dual;

/// Small exact integers, so a float scan compares exactly against the oracle.
/// Non-monotonic: a prefix min/max that dropped or duplicated a stage would still
/// look plausible on a sorted input.
fn primal(lane: usize) -> f32 {
    const PATTERN: [f32; 16] = [5.0, 2.0, 9.0, 2.0, 7.0, 1.0, 8.0, 3.0, 6.0, 4.0, 0.0, 9.0, 1.0, 7.0, 3.0, 8.0];
    PATTERN[lane % 16]
}

/// Deliberately anti-correlated with the primal: the lane with the smallest
/// primal does not have the smallest derivative, so a min-scan that kept the
/// wrong lane's derivative fails even though the primals still look right.
fn deriv(lane: usize, k: usize) -> f32 {
    (100 * (k + 1) + lane) as f32
}

macro_rules! check {
    ($label:expr, $v:ty, $n:literal) => {{
        type D = Dual<$v, $n>;
        const N: usize = $n;

        let lanes = <D as GenericVector>::LANES;

        // Build the vector and the matching array of scalar duals.
        let mut v = D::default();
        let mut want: Vec<Dual<f32, N>> = Vec::with_capacity(lanes);
        for lane in 0..lanes {
            let e = Dual {
                re: primal(lane),
                dual: core::array::from_fn(|k| deriv(lane, k)),
            };
            v = v.insertv(lane, e);
            want.push(e);
        }

        let got = |x: D| -> Vec<Dual<f32, N>> { (0..lanes).map(|i| x.extractv(i)).collect() };

        // --- sums: sequential accumulation over whole duals ------------------
        let mut acc = want[0];
        let fwd_sum: Vec<_> = want
            .iter()
            .enumerate()
            .map(|(i, &e)| {
                if i > 0 {
                    acc = Dual {
                        re: acc.re + e.re,
                        dual: core::array::from_fn(|k| acc.dual[k] + e.dual[k]),
                    };
                }
                acc
            })
            .collect();
        assert_eq!(got(v.prefix_sum()), fwd_sum, "{}: prefix_sum", $label);

        let mut acc = want[lanes - 1];
        let mut rev_sum = vec![acc; lanes];
        for i in (0..lanes - 1).rev() {
            acc = Dual {
                re: acc.re + want[i].re,
                dual: core::array::from_fn(|k| acc.dual[k] + want[i].dual[k]),
            };
            rev_sum[i] = acc;
        }
        assert_eq!(got(v.reverse_prefix_sum()), rev_sum, "{}: reverse_prefix_sum", $label);

        // --- min/max: the WHOLE winning dual, selected by primal -------------
        // `min`/`max` are `cmp_lt`/`cmp_gt` + select, which keep `self` on a tie,
        // so the oracle keeps the running accumulator on `==` to match.
        let mut acc = want[0];
        let fwd_min: Vec<_> = want
            .iter()
            .enumerate()
            .map(|(i, &e)| {
                if i > 0 && e.re < acc.re {
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
                if i > 0 && e.re > acc.re {
                    acc = e;
                }
                acc
            })
            .collect();
        assert_eq!(got(v.prefix_max()), fwd_max, "{}: prefix_max", $label);

        let mut acc = want[lanes - 1];
        let mut rev_min = vec![acc; lanes];
        for i in (0..lanes - 1).rev() {
            if want[i].re < acc.re {
                acc = want[i];
            }
            rev_min[i] = acc;
        }
        assert_eq!(got(v.reverse_prefix_min()), rev_min, "{}: reverse_prefix_min", $label);

        let mut acc = want[lanes - 1];
        let mut rev_max = vec![acc; lanes];
        for i in (0..lanes - 1).rev() {
            if want[i].re > acc.re {
                acc = want[i];
            }
            rev_max[i] = acc;
        }
        assert_eq!(got(v.reverse_prefix_max()), rev_max, "{}: reverse_prefix_max", $label);

        // --- compress / expand over every mask pattern we can afford ---------
        let patterns: Vec<u64> = if lanes <= 8 {
            (0..(1u64 << lanes)).collect()
        } else {
            let mut s = 0x9E3779B97F4A7C15u64;
            (0..512)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    s
                })
                .chain([0, u64::MAX, 0x5555_5555_5555_5555, 0xAAAA_AAAA_AAAA_AAAA])
                .collect()
        };

        // A background distinct from `v` in every lane and component, so a lane
        // taken from the wrong source is visible.
        let mut src = D::default();
        for lane in 0..lanes {
            src = src.insertv(
                lane,
                Dual {
                    re: -1.0 - lane as f32,
                    dual: core::array::from_fn(|k| -deriv(lane, k)),
                },
            );
        }

        for bits in patterns {
            // Mask from the primal: lane set <=> bit set.
            let mut sel = D::default();
            for lane in 0..lanes {
                let on = (bits >> (lane % 64)) & 1 == 1;
                sel = sel.insertv(lane, Dual { re: if on { 1.0 } else { 0.0 }, dual: [0.0; N] });
            }
            let mask = sel.cmp_gt(D::ZERO);

            // The defining law: the plain forms are mutual inverses.
            assert_eq!(got(v.compress(mask).expand(mask)), want, "{}: expand(compress) bits={bits:b}", $label);
            assert_eq!(got(v.expand(mask).compress(mask)), want, "{}: compress(expand) bits={bits:b}", $label);

            // The masked forms compose off the plain one.
            assert_eq!(
                got(v.expand_z(mask)),
                got(mask.select(v.expand(mask), D::ZERO)),
                "{}: expand_z bits={bits:b}",
                $label
            );
            assert_eq!(
                got(v.expand_m(src, mask)),
                got(mask.select(v.expand(mask), src)),
                "{}: expand_m bits={bits:b}",
                $label
            );

            // compress_m: packed selected values below the count, src's own lanes
            // at and above it. Oracle over whole duals.
            let count = (0..lanes).filter(|&l| (bits >> (l % 64)) & 1 == 1).count();
            let selected: Vec<_> = (0..lanes).filter(|&l| (bits >> (l % 64)) & 1 == 1).map(|l| want[l]).collect();
            let src_arr = got(src);
            let cm: Vec<_> = (0..lanes)
                .map(|i| if i < count { selected[i] } else { src_arr[i] })
                .collect();
            assert_eq!(got(v.compress_m(src, mask)), cm, "{}: compress_m bits={bits:b}", $label);

            // The wavefront round trip.
            assert_eq!(
                got(v.compress_z(mask).expand_m(src, mask)),
                got(mask.select(v, src)),
                "{}: compress_z + expand_m bits={bits:b}",
                $label
            );
        }
    }};
}

/// `Dual` must report its inner vector's align capability, in both directions:
/// the scans are an `align` ladder, and a wrong answer here picks the wrong
/// lowering without changing any result.
#[test]
fn forwards_native_align() {
    assert!(
        !<Dual<Vector<f32>, 2> as GenericVector>::HAS_NATIVE_ALIGN,
        "scalar backend has no native align; Dual must not claim one"
    );

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        use thermite::backend::x86_v3::f32x8;
        assert!(
            <Dual<f32x8, 2> as GenericVector>::HAS_NATIVE_ALIGN,
            "f32x8 has a native align on AVX2; Dual dropped it"
        );
    }
}

/// 1-lane scalar backend: every ladder stage is compiled out, so this pins the
/// degenerate case where a scan is the identity.
#[test]
fn scalar() {
    check!("scalar Dual<f32, 0>", Vector<f32>, 0);
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
        // ArrayRegister on AVX2: four ladder stages, and the per-chunk align path.
        check!("x86_v3 Dual<f32x16, 2>", f32x16, 2);
    }

    #[test]
    fn v2() {
        use thermite::backend::x86_v2::{f32x4, f32x8};

        check!("x86_v2 Dual<f32x4, 1>", f32x4, 1);
        check!("x86_v2 Dual<f32x8, 2>", f32x8, 2);
    }
}
