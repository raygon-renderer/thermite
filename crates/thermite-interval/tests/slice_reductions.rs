//! The slice-taking reductions on `Interval`, which is the case that decided their design.
//!
//! `hypot_s`, `inv_hypot`, `inv_sum_inv`, `harmonic_mean` and `logsumexp` have no
//! interval-specific bodies. They are default implementations in `thermite` that fold the
//! *const-length* kernel over fixed-size chunks, and `Interval` overrides those with
//! enclosing forms - so the slice forms inherit the enclosure for free, and would silently
//! lose it if they had been written directly against the values instead.
//!
//! These tests are that inheritance, asserted: an enclosure that contains the true value,
//! at lengths on both sides of the chunk boundary. A slice form written the other way
//! passes nothing here.

use thermite::math::{CoreMath as _, RealMath as _, SpatialMath as _};
use thermite::prelude::*;
use thermite_interval::{Interval, Tightest, WideningPolicy};

type V1 = Vector<f64>;
type I<W> = Interval<V1, W>;

fn iv<W: WideningPolicy>(lo: f64, hi: f64) -> I<W> {
    Interval::bounds(V1::splat(lo), V1::splat(hi))
}

fn bounds<W: WideningPolicy>(i: I<W>) -> (f64, f64) {
    (i.lo().extract::<0>(), i.hi().extract::<0>())
}

/// Two lengths, one short and one long enough that a loop-carried accumulator has
/// somewhere to drift.
fn widths() -> [usize; 2] {
    [3, 11]
}

#[test]
fn slice_hypot_encloses() {
    for n in widths() {
        // Each element is [k, k+1], so the true norm lies between the norm of the lower
        // endpoints and the norm of the upper ones.
        let xs: Vec<I<Tightest>> = (0..n).map(|k| iv(k as f64, k as f64 + 1.0)).collect();

        let (lo, hi) = bounds(I::<Tightest>::hypot_s(&xs));

        let want_lo = (0..n).map(|k| (k * k) as f64).sum::<f64>().sqrt();
        let want_hi = (0..n).map(|k| ((k + 1) * (k + 1)) as f64).sum::<f64>().sqrt();

        assert!(
            lo <= want_lo + 1e-12 && hi >= want_hi - 1e-12,
            "hypot_s over {n} intervals: [{lo}, {hi}] does not enclose [{want_lo}, {want_hi}]"
        );
    }
}

#[test]
fn slice_inv_hypot_encloses() {
    for n in widths() {
        let xs: Vec<I<Tightest>> = (0..n).map(|k| iv(k as f64 + 1.0, k as f64 + 2.0)).collect();

        let (lo, hi) = bounds(I::<Tightest>::inv_hypot(&xs));

        // Decreasing in every |x_i|, so the bounds swap.
        let want_hi = 1.0 / (0..n).map(|k| ((k + 1) * (k + 1)) as f64).sum::<f64>().sqrt();
        let want_lo = 1.0 / (0..n).map(|k| ((k + 2) * (k + 2)) as f64).sum::<f64>().sqrt();

        assert!(
            lo <= want_lo + 1e-12 && hi >= want_hi - 1e-12,
            "inv_hypot over {n} intervals: [{lo}, {hi}] does not enclose [{want_lo}, {want_hi}]"
        );
    }
}

#[test]
fn slice_logsumexp_encloses() {
    for n in widths() {
        let xs: Vec<I<Tightest>> = (0..n).map(|k| iv(k as f64, k as f64 + 0.5)).collect();

        let (lo, hi) = bounds(I::<Tightest>::logsumexp(&xs));

        let want_lo = (0..n).map(|k| (k as f64).exp()).sum::<f64>().ln();
        let want_hi = (0..n).map(|k| (k as f64 + 0.5).exp()).sum::<f64>().ln();

        assert!(
            lo <= want_lo + 1e-12 && hi >= want_hi - 1e-12,
            "logsumexp over {n} intervals: [{lo}, {hi}] does not enclose [{want_lo}, {want_hi}]"
        );
    }
}

#[test]
fn slice_reciprocal_sums_enclose() {
    for n in widths() {
        let xs: Vec<I<Tightest>> = (0..n).map(|k| iv(k as f64 + 1.0, k as f64 + 1.5)).collect();

        let (lo, hi) = bounds(I::<Tightest>::inv_sum_inv(&xs));

        // Increasing in every x_i.
        let want_lo = 1.0 / (0..n).map(|k| 1.0 / (k as f64 + 1.0)).sum::<f64>();
        let want_hi = 1.0 / (0..n).map(|k| 1.0 / (k as f64 + 1.5)).sum::<f64>();

        assert!(
            lo <= want_lo + 1e-12 && hi >= want_hi - 1e-12,
            "inv_sum_inv over {n} intervals: [{lo}, {hi}] does not enclose [{want_lo}, {want_hi}]"
        );

        let (lo, hi) = bounds(I::<Tightest>::harmonic_mean(&xs));

        assert!(
            lo <= want_lo * n as f64 + 1e-12 && hi >= want_hi * n as f64 - 1e-12,
            "harmonic_mean over {n} intervals: [{lo}, {hi}] does not enclose the scaled bounds"
        );
    }
}
