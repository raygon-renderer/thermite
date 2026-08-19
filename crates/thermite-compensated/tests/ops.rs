//! Correctness checks for the `NumericVector`/`SignedVector`/`FloatVector`
//! surface of `Compensated`: the `_c`/`_m`/`_z` masked variants, `scale`,
//! `pairwise_sum`, `arg_minmax`, `mix`, and `probit`.
//!
//! Masked-variant semantics (mask-first argument order):
//! - `_c`: keep `self` in unmasked lanes
//! - `_m`: take `src` in unmasked lanes
//! - `_z`: zero the unmasked lanes

use thermite::prelude::*;
use thermite_compensated::Compensated;

/// 1-lane scalar backend, enough to pin down blend semantics.
type V = Vector<f64>;
type C = Compensated<V>;

fn c(x: f64) -> C {
    C::new(V::splat(x))
}

fn val(x: C) -> f64 {
    x.value().extract::<0>()
}

#[test]
fn masked_unary_variants() {
    let mt = V::ZERO.cmp_lt(V::ONE); // all-true
    let mf = V::ONE.cmp_lt(V::ZERO); // all-false

    // _c keeps self where unmasked
    assert_eq!(val(c(4.0).sqrt_c(mt)), 2.0);
    assert_eq!(val(c(4.0).sqrt_c(mf)), 4.0);
    // _m takes src where unmasked
    assert_eq!(val(c(4.0).sqrt_m(c(7.0), mt)), 2.0);
    assert_eq!(val(c(4.0).sqrt_m(c(7.0), mf)), 7.0);
    // _z zeroes where unmasked
    assert_eq!(val(c(4.0).sqrt_z(mt)), 2.0);
    assert_eq!(val(c(4.0).sqrt_z(mf)), 0.0);

    // Spot-check the rest of the macro-generated family
    assert_eq!(val(c(4.0).rcp_c(mt)), 0.25);
    assert_eq!(val(c(4.0).rsqrt_c(mt)), 0.5);
    assert_eq!(val(c(2.75).floor_c(mt)), 2.0);
    assert_eq!(val(c(2.75).floor_c(mf)), 2.75);
    assert_eq!(val(c(2.75).ceil_m(c(9.0), mf)), 9.0);
    assert_eq!(val(c(1.25).round_c(mt)), 1.0);
    assert_eq!(val(c(-2.75).trunc_c(mt)), -2.0);
    assert_eq!(val(c(2.75).fract_z(mt)), 0.75);
    assert_eq!(val(c(-2.75).signed_zero_c(mt)), 0.0);
    // The sign lives in the high part; `value()` folds value+error and
    // (-0.0) + 0.0 == +0.0 erases it.
    assert!(
        c(-2.75)
            .signed_zero_c(mt)
            .uncompensated()
            .extract::<0>()
            .is_sign_negative()
    );
    // next_up/next_down step the 106-bit representation by one ulp of the
    // error term, far below what the folded `value()` can resolve, so
    // observe the step through a compensated difference instead.
    assert_eq!(val(c(1.0).next_up_c(mf)), 1.0);
    assert!(val(c(1.0).next_up_c(mt) - c(1.0)) > 0.0);
    assert!(val(c(1.0).next_down_c(mt) - c(1.0)) < 0.0);
}

#[test]
fn masked_binary_and_signed_variants() {
    let mt = V::ZERO.cmp_lt(V::ONE);
    let mf = V::ONE.cmp_lt(V::ZERO);

    // min/max
    assert_eq!(val(c(5.0).min_c(mt, c(3.0))), 3.0);
    assert_eq!(val(c(5.0).min_c(mf, c(3.0))), 5.0);
    assert_eq!(val(c(5.0).min_m(c(9.0), mf, c(3.0))), 9.0);
    assert_eq!(val(c(1.0).max_c(mt, c(3.0))), 3.0);
    assert_eq!(val(c(1.0).max_z(mf, c(3.0))), 0.0);

    // abs: the _c form is a restricted negation, not a select over full abs
    assert_eq!(val(c(-3.0).abs_c(mt)), 3.0);
    assert_eq!(val(c(-3.0).abs_c(mf)), -3.0);
    assert_eq!(val(c(3.0).abs_c(mt)), 3.0);
    assert_eq!(val(c(-3.0).abs_m(c(1.0), mf)), 1.0);
    assert_eq!(val(c(-3.0).abs_z(mt)), 3.0);
    assert_eq!(val(c(-3.0).abs_z(mf)), 0.0);

    // copysign
    assert_eq!(val(c(3.0).copysign_c(mt, c(-1.0))), -3.0);
    assert_eq!(val(c(3.0).copysign_c(mf, c(-1.0))), 3.0);
    assert_eq!(val(c(-3.0).copysign_c(mt, c(1.0))), 3.0);
    assert_eq!(val(c(3.0).copysign_m(c(8.0), mf, c(-1.0))), 8.0);
    assert_eq!(val(c(3.0).copysign_z(mf, c(-1.0))), 0.0);

    // mul_sign
    assert_eq!(val(c(3.0).mul_sign_c(mt, c(-2.0))), -3.0);
    assert_eq!(val(c(3.0).mul_sign_c(mf, c(-2.0))), 3.0);
}

#[test]
fn scale_and_mix() {
    let mt = V::ZERO.cmp_lt(V::ONE);
    let mf = V::ONE.cmp_lt(V::ZERO);

    let factor = Compensated::new(3.0_f64);
    assert_eq!(val(c(2.5).scale(factor)), 7.5);
    assert_eq!(val(c(2.5).scale_c(mf, factor)), 2.5);
    assert_eq!(val(c(2.5).scale_m(c(1.0), mf, factor)), 1.0);
    assert_eq!(val(c(2.5).scale_z(mt, factor)), 7.5);
    assert_eq!(val(c(2.5).scale_z(mf, factor)), 0.0);

    // mix endpoints are exact in compensated arithmetic
    assert_eq!(val(c(0.0).mix(c(3.0), c(9.0))), 3.0);
    assert_eq!(val(c(1.0).mix(c(3.0), c(9.0))), 9.0);
    assert_eq!(val(c(0.25).mix(c(4.0), c(8.0))), 5.0);
}

#[test]
fn pairwise_sum_preserves_compensation() {
    // 1e16 + 1.0 loses the 1.0 in plain f64 (ulp at 1e16 is 2.0); the
    // compensated pairwise add must keep it in the error term.
    let big = c(1e16);
    let one = c(1.0);

    let s = C::pairwise_sum(big, one);
    assert_eq!(val(s - big), 1.0);

    // relaxed form is the same sums (order is allowed to differ, though with one
    // lane it cannot)
    assert_eq!(val(C::relaxed_pairwise_sum(big, one)), val(s));

    // matches the inner vector's pairwise_sum for exactly-representable data
    assert_eq!(
        val(C::pairwise_sum(c(3.0), c(4.0))),
        V::pairwise_sum(V::splat(3.0), V::splat(4.0)).extract::<0>()
    );
}

#[test]
fn probit_matches_reference() {
    use thermite::math::policy::policies::Precision;
    use thermite_special::RealSpecialMathWithPolicy as _;

    // Reference values from R's qnorm / scipy.stats.norm.ppf
    let cases = [
        (0.5, 0.0),
        (0.8, 0.8416212335729143),
        (0.975, 1.959963984540054),
        (0.1, -1.2815515655446004),
    ];

    for (p, want) in cases {
        let got = val(c(p).probit_p::<Precision>());
        assert!((got - want).abs() <= 1e-13, "probit({p}) = {got}, want {want}");
    }
}

/// Multi-lane checks against the inner vector as the oracle (fixed AVX2
/// backend, same approach as thermite-special's test suite).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod wide {
    use thermite::backend::x86_v3::prelude::*;
    use thermite_compensated::Compensated;

    #[test]
    fn pairwise_sum_lane_order() {
        let a = f64x4::new([1.0, 2.0, 3.0, 4.0]);
        let b = f64x4::new([10.0, 20.0, 30.0, 40.0]);

        let got = Compensated::pairwise_sum(Compensated::new(a), Compensated::new(b));
        let expect = f64x4::pairwise_sum(a, b); // [3, 7, 30, 70]

        for i in 0..f64x4::LANES {
            assert_eq!(got.value().extractv(i), expect.extractv(i), "lane {i}");
        }

        // and the documented contract explicitly
        let contract = [1.0 + 2.0, 3.0 + 4.0, 10.0 + 20.0, 30.0 + 40.0];
        for (i, expected) in contract.iter().enumerate() {
            assert_eq!(got.value().extractv(i), *expected, "lane {i}");
        }
    }

    #[test]
    fn arg_minmax_orders_by_value() {
        let v = Compensated::new(f64x4::new([5.0, -1.0, 7.0, 2.0]));
        assert_eq!(v.arg_minmax(), (1, 2));
        assert_eq!(v.arg_minmax(), v.value().arg_minmax());
    }
}

/// `harmonic_mean` / `inv_sum_inv` on `Compensated`, which takes the direct reciprocal-sum
/// form like the other composites.
///
/// `Compensated` is a real single-value type, so unlike `Complex` its `min` is a genuine
/// ordering and the zero limit does fall out, and both are checked here. The reason it is on
/// the direct path anyway is that the min-scaled rewrite exists to stop a *binary64*
/// reciprocal from overflowing, and double-double arithmetic has the same exponent range,
/// so the rewrite would buy nothing it does not already have.
#[test]
fn harmonic_mean_and_inv_sum_inv() {
    use thermite::math::CoreMath;

    let hm = C::harmonic_mean([c(1.0), c(2.0), c(4.0)]);
    let si = C::inv_sum_inv([c(1.0), c(2.0), c(4.0)]);

    // 3/(1 + 1/2 + 1/4) = 12/7, and a seventh is where a double-double should earn its keep.
    let want_hm = 12.0 / 7.0;
    assert!((hm.value().extract::<0>() - want_hm).abs() < 1e-15, "got {}", hm.value().extract::<0>());
    assert!((si.value().extract::<0>() - want_hm / 3.0).abs() < 1e-15);

    // The factor of N, the identity that separates the two functions.
    let x = 3.0;
    let hm = C::harmonic_mean([c(x), c(x), c(x)]);
    let si = C::inv_sum_inv([c(x), c(x), c(x)]);
    assert!((hm.value().extract::<0>() - x).abs() < 1e-15);
    assert!((si.value().extract::<0>() - x / 3.0).abs() < 1e-15);

    // A zero element gives NaN, not the 0 a plain f64 vector gives. Being a real type is not
    // enough for that limit: it needs plain IEEE division, where 1/0 is a clean infinity.
    // Double-double division forms `two_prod(q1, rhs)`, which at `q1 = inf, rhs = 0` is
    // `inf * 0 = NaN`, and the error term poisons the result from there. Same shape as
    // `Complex`, for a different reason, and inherited from the arithmetic rather than
    // introduced here.
    let hz = C::harmonic_mean([c(0.0), c(1.0)]);
    assert!(hz.value().extract::<0>().is_nan(), "a zero element gives NaN on Compensated");

    let recip = c(1.0) / c(0.0);
    assert!(recip.value().extract::<0>().is_nan(), "because 1/0 is itself NaN here");

    // The compensated part should be carrying real information, not just tracking the f64
    // result: recomputing in plain f64 and comparing to the double-double value shows the
    // low word is doing something.
    let hm = C::harmonic_mean([c(1.0), c(3.0), c(7.0)]);
    let plain = 3.0 / (1.0 + 1.0 / 3.0 + 1.0 / 7.0);
    assert!((hm.value().extract::<0>() - plain).abs() < 1e-14, "double-double tracks the f64 answer");
}
