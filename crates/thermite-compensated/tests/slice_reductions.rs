//! The slice-taking reductions on a COMPOSITE, which is the only thing that reaches the
//! trait defaults.
//!
//! `thermite`'s own `tests/slice_reductions.rs` runs against `Vector<f32>`/`Vector<f64>`,
//! and those route past those defaults entirely: `specialized::ps`/`pd` override `hypot_s`,
//! `inv_hypot`, `inv_sum_inv` and `harmonic_mean` with the `generic::*_slice_internal`
//! bodies. Every real-vector assertion there is therefore about code this file cannot
//! reach, and vice versa.
//!
//! `Compensated` is the cleanest vehicle: it overrides neither the const kernels nor the
//! slice forms, so `hypot_s` here is `hypot_slice_recip_scaled`, the default. `Dual` and `Interval`
//! do override, and are covered in their own crates.
//!
//! Lengths run well past anything a const form is instantiated at, since a loop-carried
//! accumulator is the new machinery and a short slice never exercises it.

use thermite::prelude::*;
use thermite_compensated::Compensated;

/// 1-lane scalar backend: the fold is lane-invariant, so width buys nothing here.
type V = Vector<f64>;
type C = Compensated<V>;

fn c(x: f64) -> C {
    C::new(V::splat(x))
}

fn val(x: C) -> f64 {
    x.value().extract::<0>()
}

fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    }
}

/// The slice form is a serial fold where the const form is a log-depth tree, so the two
/// associativity orders are only guaranteed to agree bit for bit when the intermediate sums
/// are exact. Powers of two make them exact, which is why this can assert equality rather
/// than a tolerance, and equality is the sharper assertion where it is available.
#[test]
fn short_slice_matches_the_const_form() {
    let arr = [c(1.0), c(2.0), c(4.0), c(8.0), c(16.0)];

    assert_eq!(
        val(C::hypot_s(&arr)),
        val(C::hypot_n(arr)),
        "hypot_s disagrees with hypot_n"
    );

    assert_eq!(
        val(C::inv_hypot(&arr)),
        val(C::inv_hypot_n(arr)),
        "inv_hypot disagrees with inv_hypot_n"
    );

    assert_eq!(
        val(C::logsumexp(&arr)),
        val(C::logsumexp_n(arr)),
        "logsumexp disagrees with logsumexp_n"
    );

    assert_eq!(
        val(C::inv_sum_inv(&arr)),
        val(C::inv_sum_inv_n(arr)),
        "inv_sum_inv disagrees with inv_sum_inv_n"
    );

    assert_eq!(
        val(C::harmonic_mean(&arr)),
        val(C::harmonic_mean_n(arr)),
        "harmonic_mean disagrees with harmonic_mean_n"
    );
}

/// The regression this file exists for.
///
/// `inv_hypot` scales before taking the root and so never forms the norm, which gives it a
/// strictly wider output range than the reciprocal of `hypot_s` has: at these magnitudes the
/// norm itself saturates while its inverse is perfectly representable. Any implementation
/// that computes the norm and then inverts it (which the chunk-folding one did, and which
/// is the obvious way to write it) answers `0` here instead.
#[test]
fn inv_hypot_keeps_the_const_form_range() {
    // 1.5e308 * sqrt(3) = 2.598e308, which does not fit in a binary64 exponent.
    let arr = [c(1.5e308), c(1.5e308), c(1.5e308)];

    // Not merely infinite: a double-double that overflows takes `inf - inf` in its error
    // term, so `Compensated` saturates to NaN rather than to a signed infinity. Either way
    // the norm is unusable and its reciprocal cannot be the answer.
    let forward = val(C::hypot_s(&arr));
    assert!(
        !forward.is_finite(),
        "the forward norm is expected to saturate here; got {forward}"
    );
    assert!(!(1.0 / forward).is_normal(), "so its reciprocal is the wrong answer");

    let inv = val(C::inv_hypot(&arr));
    let want = val(C::inv_hypot_n(arr));

    assert!(
        inv > 0.0 && inv.is_finite(),
        "inv_hypot saturated to {inv} where the const form gives {want}"
    );

    // Subnormal, so the agreement is coarse by construction. The point is that both are
    // the same small number rather than one of them being zero.
    assert!(rel(inv, want) < 1e-9, "inv_hypot = {inv}, inv_hypot_n = {want}");
}

/// Lengths well past any const instantiation, against closed forms rather than against the
/// const kernel (which cannot take a runtime length).
#[test]
fn many_lengths_agree_with_closed_forms() {
    for n in [1usize, 7, 8, 9, 16, 63, 64, 65, 100, 129] {
        let ones = vec![c(1.0); n];

        // ||[1; n]|| = sqrt(n)
        let got = val(C::hypot_s(&ones));
        let want = (n as f64).sqrt();
        assert!(rel(got, want) < 1e-14, "hypot_s(n = {n}) = {got}, want {want}");

        // 1/||[1; n]|| = 1/sqrt(n)
        let got = val(C::inv_hypot(&ones));
        let want = 1.0 / (n as f64).sqrt();
        assert!(rel(got, want) < 1e-9, "inv_hypot(n = {n}) = {got}, want {want}");

        // 1/(sum of n reciprocals of 1) = 1/n
        let got = val(C::inv_sum_inv(&ones));
        let want = 1.0 / n as f64;
        assert!(rel(got, want) < 1e-14, "inv_sum_inv(n = {n}) = {got}, want {want}");

        // n/(sum of n reciprocals of 1) = 1
        let got = val(C::harmonic_mean(&ones));
        assert!(rel(got, 1.0) < 1e-14, "harmonic_mean(n = {n}) = {got}, want 1");

        // ln(sum of n copies of e^0) = ln(n)
        let zeros = vec![c(0.0); n];
        let got = val(C::logsumexp(&zeros));
        let want = (n as f64).ln();
        assert!(rel(got, want) < 1e-13, "logsumexp(n = {n}) = {got}, want {want}");
    }
}

/// A spread wide enough that an unscaled sum of squares would overflow at the top and lose
/// the bottom entirely, carried the length of the accumulation loop.
#[test]
fn wide_spread_keeps_the_scaling() {
    let mut arr = vec![c(1e-160); 12];
    arr[0] = c(1e160);

    // sqrt(1e320 + 11e-320) = 1e160 exactly at binary64 resolution, and the direct
    // (unscaled) form would square 1e160 into an overflow.
    let got = val(C::hypot_s(&arr));
    assert!(rel(got, 1e160) < 1e-14, "hypot_s = {got}, want 1e160");

    let got = val(C::inv_hypot(&arr));
    assert!(rel(got, 1e-160) < 1e-9, "inv_hypot = {got}, want 1e-160");
}

/// The empty slice, whose `hypot` answer is the one case that depends on the direction.
#[test]
fn empty_slice_limits() {
    let none: [C; 0] = [];

    assert_eq!(val(C::hypot_s(&none)), 0.0, "the empty norm is 0");
    assert!(
        val(C::inv_hypot(&none)).is_infinite(),
        "and 1/0 is infinity, matching inv_hypot_n at N = 0"
    );
    assert!(
        val(C::logsumexp(&none)).is_infinite() && val(C::logsumexp(&none)) < 0.0,
        "the empty log-sum-exp is ln(0) = -inf"
    );
    // The empty sum of reciprocals is 0, so the answer is this type's own `1/0`, which on
    // a double-double is NaN, not infinity, because division forms `two_prod(q1, rhs)` and
    // that is `inf * 0` at a zero divisor. Inherited from the arithmetic rather than chosen
    // here, and pinned the same way in `ops.rs`.
    let empty = val(C::inv_sum_inv(&none));
    let one_over_zero = val(c(1.0) / c(0.0));
    assert!(
        empty.is_nan() && one_over_zero.is_nan(),
        "the empty reciprocal sum is this type's 1/0; got {empty} against {one_over_zero}"
    );
}
