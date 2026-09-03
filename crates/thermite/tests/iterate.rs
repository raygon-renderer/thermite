//! Smoke gate for the three iteration drivers in `math::algorithms::iterate`.
//!
//! Each driver is checked against a target with a _known closed form_, so a failure points at
//! the driver rather than at whatever kernel is riding on it. The per-lane test matters most:
//! these run at a real register width, because `Vector<f64>` is the one-lane scalar seed and a
//! packet test written against it is a single lane that proves nothing while passing.
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use thermite::Vector;
use thermite::backend::x86_v2::X86V2;
use thermite::math::algorithms::{lentz, prod_f, sum_counted, sum_f, sum_pair, sum_ratio};
use thermite::math::policy::policies::Precision;
use thermite::prelude::*;

type V = Vector<f64>;
/// A real register width, so the per-lane behaviour is actually exercised.
type W = Vector<<X86V2 as Simd>::f64x2>;

fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        return if got == 0.0 { 0.0 } else { f64::INFINITY };
    }
    ((got - want) / want).abs()
}

/// `phi = 1 + 1/(1 + 1/(1 + ...))`, every `a_j` and `b_j` equal to one.
///
/// The slowest-converging continued fraction there is (its convergents are ratios of
/// consecutive Fibonacci numbers), so it exercises the iteration cap as well as the value.
#[test]
fn lentz_finds_the_golden_ratio() {
    let phi = (1.0 + 5.0f64.sqrt()) / 2.0;
    let tol = V::splat(1e-15);

    let got = lentz::<V, Precision, _>(tol, GenericMask::TRUTHY, V::ONE, |_j| (V::ONE, V::ONE));

    let value = got.expect("golden-ratio fraction should converge").extract::<0>();
    assert!(rel(value, phi) <= 1e-14, "lentz gave {value}, want {phi}");
}

/// `e^x = sum x^k / k!`, advanced by the ratio `t_{k+1} = t_k * x/(k+1)`.
///
/// All terms share a sign for positive `x`, so this checks the driver rather than any
/// cancellation behaviour.
#[test]
fn sum_ratio_reproduces_exp() {
    let tol = V::splat(1e-16);

    for &x in &[0.25f64, 1.0, 2.5, 8.0] {
        let xv = V::splat(x);
        let got = sum_ratio::<V, Precision, _>(tol, GenericMask::TRUTHY, V::ONE, |k, term| {
            term * xv / V::splat(k as f64)
        });

        let value = got.expect("exp series should converge").extract::<0>();
        assert!(rel(value, x.exp()) <= 1e-14, "exp({x}): got {value}, want {}", x.exp());
    }
}

/// The counted driver has no test at all, so at a term count past convergence it must agree
/// with the converging one bit for bit.
#[test]
fn sum_counted_agrees_once_past_convergence() {
    let tol = V::splat(1e-16);
    let xv = V::splat(1.5);

    let advance = |k: i64, term: V| term * xv / V::splat(k as f64);

    let converged = sum_ratio::<V, Precision, _>(tol, GenericMask::TRUTHY, V::ONE, advance)
        .expect("exp series should converge")
        .extract::<0>();
    // 40 terms is far past where `x = 1.5` stops moving the sum.
    let counted = sum_counted::<V, Precision, 40, _>(V::ONE, advance).extract::<0>();

    assert_eq!(
        converged.to_bits(),
        counted.to_bits(),
        "counted {counted} and converged {converged} must be identical past convergence"
    );
}

/// Lanes that converge at different rates must each keep their own answer.
///
/// This is the property the `select` in `lentz` exists for: the fast lane converges first and
/// must then be held, while the slow lane keeps iterating. Without the freeze, the fast lane
/// would keep being multiplied by a delta that is only approximately one.
#[test]
fn lentz_freezes_lanes_independently() {
    // `sqrt(1+z) - 1` style fraction: b0 = 1, a_j = z, b_j = 2. Converges to `sqrt(1+z)` for
    // z > 0, and fast for small z, slowly for large.
    let z = W::splat(0.01).insert::<1>(3.0);
    let tol = W::splat(1e-15);

    let got = lentz::<W, Precision, _>(tol, GenericMask::TRUTHY, W::ONE, |_j| (z, W::ONE + W::ONE));
    let f = got.expect("both lanes should converge");

    for (i, zi) in [0.01f64, 3.0].iter().enumerate() {
        let want = (1.0 + zi).sqrt();
        let value = f.extractv(i);
        assert!(
            rel(value, want) <= 1e-13,
            "lane {i} (z = {zi}): got {value}, want {want}"
        );
    }
}

/// The tolerance is relative to the largest term, and the term that trips convergence is kept.
///
/// Both are changes from the previous `sum_f`. This series fails under the old behaviour in
/// the most misleading way available: it returns success with the wrong answer.
///
/// Terms are `1e-20 * 0.5^n`, summing to `2e-20`. An **absolute** tolerance of `1e-15` is
/// already satisfied by the very first term, so the old code stopped immediately. Because
/// it tested before accumulating, it discarded that term too and returned `Ok(0.0)`.
#[test]
fn sum_f_tolerance_is_relative_to_the_largest_term() {
    let tol = V::splat(1e-15);
    let tiny = |n: i64| V::splat(1e-20 * 0.5f64.powi(n as i32));

    let got = sum_f::<V, Precision, _>(tol, GenericMask::TRUTHY, 0, 200, tiny);
    let value = got
        .expect("scaled-down geometric series should converge")
        .extract::<0>();

    assert!(
        rel(value, 2e-20) <= 1e-12,
        "got {value}, want 2e-20 - an absolute tolerance would have returned 0.0 here"
    );
    assert!(value != 0.0, "the first term must not be discarded");
}

/// A series shorter than one check stride must still be able to report convergence.
///
/// The reduction is amortized over four iterations, so without the explicit final-iteration
/// test a two-term series would converge and still be reported as a failure.
#[test]
fn sum_f_converges_inside_one_check_stride() {
    let tol = V::splat(1e-3);
    // 1, then 1e-9: the second term is far below tolerance relative to the first.
    let two = |n: i64| if n == 0 { V::ONE } else { V::splat(1e-9) };

    let got = sum_f::<V, Precision, _>(tol, GenericMask::TRUTHY, 0, 2, two);
    assert!(got.is_ok(), "a two-term series must be able to converge");
}

/// `prod_f`'s tolerance applies to the FACTOR, not to the change in the product.
///
/// Under the old absolute test on `new_prod - prod`, convergence depended on how large the
/// product happened to be. Scaling the whole product down by `1e-12` made the same factors
/// converge instantly. The delta was tiny because the product was tiny, not because the
/// factors had settled.
#[test]
fn prod_f_tolerance_applies_to_the_factor() {
    let tol = V::splat(1e-9);

    // Factors 1 + 0.5^(n+1), but the whole product pre-scaled to be minuscule. The factors are
    // identical in both runs, so the converged answers must agree to the tolerance.
    let factors = |n: i64| V::splat(1.0 + 0.5f64.powi(n as i32 + 1));
    let plain = prod_f::<V, Precision, _>(tol, GenericMask::TRUTHY, 0, 80, factors);

    let scaled = prod_f::<V, Precision, _>(tol, GenericMask::TRUTHY, 0, 80, |n| {
        if n == 0 {
            factors(0) * V::splat(1e-12)
        } else {
            factors(n)
        }
    });

    let a = plain.expect("converging product").extract::<0>();
    let b = scaled.expect("scaled converging product").extract::<0>();

    assert!(
        rel(b, a * 1e-12) <= 1e-7,
        "scaling the product must not change where it converges: {a} vs {b}"
    );
}

/// `prod_f` keeps the factor that trips convergence.
///
/// The old version returned the product from _before_ that factor, discarding one it had
/// already paid to compute.
#[test]
fn prod_f_keeps_the_converging_factor() {
    // Two factors: 2, then exactly 1. The second trips convergence at any tolerance and
    // multiplying it in is a no-op, so the answer must be 2 either way, but a third factor
    // makes the discard visible.
    let tol = V::splat(1e-6);
    let got = prod_f::<V, Precision, _>(tol, GenericMask::TRUTHY, 0, 2, |n| {
        if n == 0 { V::TWO } else { V::splat(1.0 + 1e-9) }
    });

    let value = got.expect("must converge on the near-one factor").extract::<0>();
    assert!(
        rel(value, 2.0 * (1.0 + 1e-9)) <= 1e-15,
        "got {value}: the converging factor must be included"
    );
}

/// An inactive lane must not be able to hold the loop open.
///
/// A packet spans regions, so a kernel that splits its domain runs every region any lane
/// needs, and a lane bound for a different arm sits in this one's loop regardless. If it gets
/// to vote on convergence it sets the trip count for everyone.
///
/// Unlike the same property inside a Bessel arm (where a masked-but-iterating lane produces
/// identical answers, so only a benchmark can catch a missing mask), here it is **directly
/// observable**: lane 1 never converges at all, so a missing mask turns `Ok` into `Err`.
#[test]
fn an_inactive_lane_cannot_hold_the_loop_open() {
    let tol = W::splat(1e-12);

    // Lane 0: geometric, converges to 2. Lane 1: constant 1, never converges.
    let f = |n: i64| W::splat(0.5f64.powi(n as i32)).insert::<1>(1.0);

    // Both lanes live: the divergent lane prevents convergence, as it should.
    let both = sum_f::<W, Precision, _>(tol, GenericMask::TRUTHY, 0, 400, f);
    assert!(both.is_err(), "a genuinely divergent lane must report failure");

    // Lane 0 only: the divergent lane is not asked about, so the sum converges.
    let first_only = W::ONE.insert::<1>(0.0).cmp_gt(W::ZERO);
    let masked = sum_f::<W, Precision, _>(tol, first_only, 0, 400, f);

    let v = masked
        .expect("an inactive lane must not prevent convergence")
        .extractv(0);
    assert!(rel(v, 2.0) <= 1e-11, "active lane summed to {v}, want 2.0");
}

/// A region no lane needs costs one reduction, not a loop.
#[test]
fn a_fully_inactive_call_short_circuits() {
    let tol = W::splat(1e-12);
    let none = <W as GenericVector>::Mask::FALSY;

    // `f` would never converge, and `lentz`'s fraction would run to the iteration cap, but
    // neither is asked to, so both report success immediately.
    assert!(sum_f::<W, Precision, _>(tol, none, 0, 400, |_| W::ONE).is_ok());
    assert!(prod_f::<W, Precision, _>(tol, none, 0, 400, |_| W::TWO).is_ok());
    assert!(lentz::<W, Precision, _>(tol, none, W::ONE, |_j| (W::ONE, W::ONE)).is_ok());
}

/// The paired driver against two series with a known closed form, sharing one chain.
///
/// `cosh` and `sinh` from `e^x`'s terms split by parity: one `x^k/k!` chain feeding two
/// accumulators, which is the shape `sum_pair` exists for.
#[test]
fn sum_pair_reproduces_cosh_and_sinh() {
    let tol = W::splat(1e-16);

    for &x in &[0.25f64, 1.0, 2.5, 6.0] {
        let xv = W::splat(x);
        let mut term = W::ONE; // x^k / k!
        let mut k = 0i64;

        // Even k -> cosh, odd k -> sinh. One chain, two destinations.
        let got = sum_pair::<W, Precision, _>(tol, GenericMask::TRUTHY, (W::ONE, W::ZERO), || {
            k += 1;
            term = term * xv / W::splat(k as f64);
            let even = k % 2 == 0;
            (if even { term } else { W::ZERO }, if even { W::ZERO } else { term })
        });

        let (c, s) = got.expect("both series should converge");
        assert!(rel(c.extractv(0), x.cosh()) <= 1e-14, "cosh({x}) = {}", c.extractv(0));
        assert!(rel(s.extractv(0), x.sinh()) <= 1e-14, "sinh({x}) = {}", s.extractv(0));
    }
}

/// Both components must converge, not just the first.
///
/// Boost's Temme series tests only its first accumulator, which is safe there and not in
/// general. Nothing makes a second component's terms shrink at the same rate. Here the second
/// never settles, so a first-component-only test would wrongly report success.
#[test]
fn sum_pair_requires_both_components() {
    let tol = W::splat(1e-12);

    let mut k = 0i64;
    let got = sum_pair::<W, Precision, _>(tol, GenericMask::TRUTHY, (W::ZERO, W::ZERO), || {
        k += 1;
        // First component converges geometrically. The second is a constant that never does.
        (W::splat(0.5f64.powi(k as i32)), W::ONE)
    });

    assert!(
        got.is_err(),
        "a non-converging second component must not report success"
    );
}
