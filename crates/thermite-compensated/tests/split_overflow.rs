//! Regression: Veltkamp's splitting must not overflow on large operands.
//!
//! `two_prod` splits with `a * SPLITTER`. That product overflows to infinity once `|a|`
//! passes `MAX / SPLITTER`, and `inf - inf` is NaN, so without a guard `two_prod` returns
//! NaN for operands whose true product is an ordinary finite number.
//!
//! This only ever affected targets WITHOUT hardware FMA, because `two_prod` takes an FMA
//! fast path when `HAS_TRUE_FMA` and never splits at all. The x86_v1 (SSE2) and x86_v2
//! (SSE4.2) backends are exactly that case, so they are what these tests pin.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use thermite_compensated::ScalarValue;

/// Overflow thresholds for `a * SPLITTER`, per type.
const F64_THRESH: f64 = 6.69692879491417e299; // 2^996
const F32_THRESH: f32 = 4.153_837_5e34; // 2^115

fn assert_exact_product_f64(a: f64, b: f64, ctx: &str) {
    let (p, e) = <f64 as ScalarValue>::two_prod(a, b);
    assert!(p.is_finite(), "{ctx}: two_prod({a:e}, {b:e}) value is {p:e}");
    assert!(e.is_finite(), "{ctx}: two_prod({a:e}, {b:e}) error is {e:e}");
    assert_eq!(p, a * b, "{ctx}: value must equal the rounded product");

    // p + e must reproduce a * b exactly, so the residual is the true rounding error.
    let recovered = a.mul_add(b, -p);
    assert_eq!(e, recovered, "{ctx}: error term is not the exact residual");
}

#[test]
fn f64_scalar_split_survives_large_operands() {
    let cases = [
        (F64_THRESH * 2.0, 3.0),
        (f64::MAX, 0.5),
        (-f64::MAX, 0.25),
        (1e300, 1e-300),
        (1.7e308, 1e-8),
        (F64_THRESH, F64_THRESH.recip()),
    ];

    for (a, b) in cases {
        assert_exact_product_f64(a, b, "f64 scalar");
        assert_exact_product_f64(b, a, "f64 scalar swapped");
    }
}

#[test]
fn f32_scalar_split_survives_large_operands() {
    for (a, b) in [
        (F32_THRESH * 2.0, 3.0f32),
        (f32::MAX, 0.5),
        (-f32::MAX, 0.25),
        (1e35, 1e-30),
    ] {
        let (p, e) = <f32 as ScalarValue>::two_prod(a, b);
        assert!(
            p.is_finite() && e.is_finite(),
            "two_prod({a:e}, {b:e}) = ({p:e}, {e:e})"
        );
        assert_eq!(p, a * b);
        assert_eq!(e, (a as f64 * b as f64 - p as f64) as f32, "error term wrong");
    }
}

/// The case the owner asked for: the same behaviour on the SSE2 backend, which has no
/// hardware FMA and therefore actually runs the splitting path.
mod sse2 {
    use super::{F32_THRESH, F64_THRESH};
    use thermite::backend::x86_v1::prelude::*;
    use thermite_compensated::ScalarValue;

    #[test]
    fn f64x2_two_prod_survives_large_operands() {
        assert!(
            !<f64x2 as thermite::vector::ops::MulAddExt>::HAS_TRUE_FMA,
            "x86_v1 must not have hardware FMA, or this test proves nothing"
        );

        for (a, b) in [
            (F64_THRESH * 2.0, 3.0),
            (f64::MAX, 0.5),
            (-f64::MAX, 0.25),
            (1.7e308, 1e-8),
        ] {
            let (p, e) = <f64x2 as ScalarValue>::two_prod(f64x2::splat(a), f64x2::splat(b));
            let (pv, ev) = (p.extract::<0>(), e.extract::<0>());

            assert!(pv.is_finite(), "two_prod({a:e}, {b:e}) value = {pv:e}");
            assert!(ev.is_finite(), "two_prod({a:e}, {b:e}) error = {ev:e}");
            assert_eq!(pv, a * b);
            assert_eq!(ev, a.mul_add(b, -pv), "error term is not the exact residual");
        }
    }

    #[test]
    fn f32x4_two_prod_survives_large_operands() {
        assert!(!<f32x4 as thermite::vector::ops::MulAddExt>::HAS_TRUE_FMA);

        for (a, b) in [(F32_THRESH * 2.0, 3.0f32), (f32::MAX, 0.5), (-f32::MAX, 0.25)] {
            let (p, e) = <f32x4 as ScalarValue>::two_prod(f32x4::splat(a), f32x4::splat(b));
            let (pv, ev) = (p.extract::<0>(), e.extract::<0>());

            assert!(pv.is_finite() && ev.is_finite(), "({pv:e}, {ev:e})");
            assert_eq!(pv, a * b);
        }
    }

    /// One huge lane must not corrupt its neighbours, since the guard is per lane, not per
    /// packet, so a mixed-magnitude vector has to come out lane-for-lane correct.
    #[test]
    fn f64x2_mixed_magnitude_lanes() {
        let a = f64x2::new([f64::MAX, 3.0]);
        let b = f64x2::new([0.5, 7.0]);

        let (p, e) = <f64x2 as ScalarValue>::two_prod(a, b);

        for (i, (av, bv)) in [(f64::MAX, 0.5), (3.0, 7.0)].into_iter().enumerate() {
            let pv = p.extractv(i);
            let ev = e.extractv(i);
            assert!(pv.is_finite() && ev.is_finite(), "lane {i}: ({pv:e}, {ev:e})");
            assert_eq!(pv, av * bv, "lane {i} value");
            assert_eq!(ev, av.mul_add(bv, -pv), "lane {i} error");
        }
    }

    /// Ordinary magnitudes must be bit-identical to the unguarded split, since both
    /// scale factors are exact powers of two and the guard does not fire.
    #[test]
    fn f64x2_small_operands_unchanged() {
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        for _ in 0..20_000 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let a = (s as i64 as f64) / 65_536.0;
            let b = (s.rotate_left(31) as i64 as f64) / 4096.0;
            if !(a * b).is_finite() {
                continue;
            }

            let (p, e) = <f64x2 as ScalarValue>::two_prod(f64x2::splat(a), f64x2::splat(b));
            assert_eq!(p.extract::<0>(), a * b);
            assert_eq!(e.extract::<0>(), a.mul_add(b, -(a * b)), "a={a:e} b={b:e}");
        }
    }
}

/// The one case the guard cannot reach, pinned so it stays visible.
///
/// Veltkamp's split rounds `hi` up by up to a relative `2^-53`, so `a_hi * b_hi`
/// slightly exceeds `a * b`. When `a * b` is within that relative distance of `MAX`,
/// that product overflows and the error term is lost. This is inherent to Dekker's
/// `two_prod` and is why the guarded version still documents a near-overflow limit.
///
/// `(MAX, 0.5)` is fine, and only a product at MAX itself is affected.
#[test]
fn f64_product_at_max_is_the_documented_limit() {
    let (p, e) = <f64 as ScalarValue>::two_prod(f64::MAX, 1.0);

    assert_eq!(p, f64::MAX, "the value is still correct");
    assert!(
        !e.is_finite(),
        "if this now returns a finite error term, the near-overflow limit was fixed -          update the docs on ScalarValue::rebalance_for_split, got {e:e}"
    );

    // One binade down is fully correct, which is what bounds the limitation.
    let (p, e) = <f64 as ScalarValue>::two_prod(f64::MAX, 0.5);
    assert_eq!(p, f64::MAX * 0.5);
    assert_eq!(e, f64::MAX.mul_add(0.5, -p));
}

/// A value sitting exactly AT the threshold takes the unguarded path, so splitting it
/// must not overflow. An earlier f32 threshold of 2^116 failed here: `2^116 * 4097`
/// exceeds `f32::MAX`, so `two_prod` at the boundary returned NaN.
#[test]
fn operands_exactly_at_the_threshold_are_safe() {
    for scale in [1.0f32, 0.5, 0.25] {
        let a = F32_THRESH * scale;
        let (p, e) = <f32 as ScalarValue>::two_prod(a, 3.0);
        assert!(
            p.is_finite() && e.is_finite(),
            "f32 two_prod({a:e}, 3.0) = ({p:e}, {e:e})"
        );
        assert_eq!(p, a * 3.0);
    }

    for scale in [1.0f64, 0.5, 0.25] {
        let a = F64_THRESH * scale;
        let (p, e) = <f64 as ScalarValue>::two_prod(a, 3.0);
        assert!(
            p.is_finite() && e.is_finite(),
            "f64 two_prod({a:e}, 3.0) = ({p:e}, {e:e})"
        );
        assert_eq!(p, a * 3.0);
        assert_eq!(e, a.mul_add(3.0, -p));
    }
}
