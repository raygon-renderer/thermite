//! `erf` and `erfc` over the right half-plane, with the wedge that the series regime
//! used to be selected into wrongly.
//!
//! The regime guard in `math/special/erf.rs` decides between a Taylor series and the
//! Faddeeva function. The series' cancellation is `e^{2 Im(z)^2}`: the terms peak near
//! `e^{|z|^2}` while the sum is `(sqrt(pi)/2) e^{z^2} erf(z)`. The guard used to test
//! `-Re(z^2) = y^2 - x^2` instead, which is the same quantity only on the imaginary axis.
//! Along a ray at 45 degrees `Re(z^2)` is exactly zero however large `z` grows, so the
//! guard never fired and the series ran at a cancellation of `e^{2y^2}`.
//!
//! What that cost, measured before the fix on `z = a(1-i)`:
//!
//! ```text
//! |z|    erf                    correct                  |erf| is <= 1.2 here
//! 5.66   (1.070e0,  1.218e-1)   (1.070e0,  1.218e-1)     ok
//! 6.36   (1.002e1,  1.501e1)    (9.062e-1, 1.323e-1)
//! 7.07   (8.467e2,  4.343e4)    (9.091e-1,-6.666e-2)
//! 8.00   (-3.020e10,-2.368e10)  (9.291e-1, 1.081e-1)
//! ```
//!
//! It was found by differencing the new real `fresnel` against
//! `C + iS = (1+i)/2 erf(sqrt(pi)(1-i)x/2)`, which walks exactly that ray (see
//! `fresnel_sici_oracle.rs`).
//!
//! The grid below deliberately sweeps the whole right half-plane rather than the ray
//! alone: the imaginary axis (where the old guard was tested and did work), both signs
//! of `Im z`, and the corners of every regime boundary.

#![cfg(feature = "special")]

use thermite::prelude::*;
use thermite_complex::Complex;
use thermite_special::{SpecialMath, SpecialMathWithPolicy};

use thermite::math::policy::DefaultPolicy;
use thermite::math::policy::policies::{BestPrecision, Performance};

type V = Vector<f64>;
type C = Complex<V>;

include!("erf_wedge_ref.rs");

fn c(re: f64, im: f64) -> C {
    Complex::new(V::splat(re), V::splat(im))
}

fn parts(z: C) -> (f64, f64) {
    (z.re.extract::<0>(), z.im.extract::<0>())
}

/// Normwise, which is the guarantee the algorithm makes. `faddeeva.rs` grades itself
/// the same way, and for the same reason: near the real axis `Re w` alone is far
/// smaller than `|w|` and carries fewer digits, which is a property of the function.
#[track_caller]
fn normwise(what: &str, x: f64, y: f64, got: (f64, f64), want: (f64, f64), tol: f64) {
    let norm = (want.0 * want.0 + want.1 * want.1).sqrt();
    let abs = ((got.0 - want.0).powi(2) + (got.1 - want.1).powi(2)).sqrt();
    // erf(0) = 0 exactly, and erfc has no zeros, so the absolute arm is only the origin.
    let err = if norm == 0.0 { abs } else { abs / norm };
    assert!(
        err <= tol,
        "{what}({x} + {y}i): got ({:e}, {:e}), want ({:e}, {:e}) - normwise {err:e} > {tol:e}",
        got.0,
        got.1,
        want.0,
        want.1
    );
}

/// Tolerances are measured over this grid, not guessed:
///
/// | | `erf` | `erfc` | at |
/// |---|---|---|---|
/// | default / `Performance` | 4.19e-10 | 5.40e-10 | `0 + 5i`, `3 + 2.5i` |
/// | `Best` | 2.89e-13 | 2.89e-13 | `0.01 + 6i` |
///
/// The default figures are the two regimes at their own limits: the `Average`-tier
/// Weideman `N`, which `erf.rs` documents as ~4e-10, and the series at the lower tiers'
/// wider cancellation budget. `Best` is the tight one and pins the fix: on
/// the 45-degree ray the old guard gave 2.3e-11 at `x = 3` and 3.9e-9 at `x = 3.5` even
/// at that tier.
const TOL_DEFAULT: f64 = 1e-8;
const TOL_BEST: f64 = 1e-12;

#[test]
fn erf_over_the_right_half_plane() {
    for &(x, y, er, ei, _, _) in ERF_REFS.iter() {
        normwise("erf", x, y, parts(c(x, y).erf()), (er, ei), TOL_DEFAULT);
    }
}

#[test]
fn erfc_over_the_right_half_plane() {
    for &(x, y, _, _, cr, ci) in ERF_REFS.iter() {
        normwise("erfc", x, y, parts(c(x, y).erfc()), (cr, ci), TOL_DEFAULT);
    }
}

#[test]
fn erf_at_best_precision() {
    for &(x, y, er, ei, _, _) in ERF_REFS.iter() {
        let got = parts(c(x, y).erf_p::<BestPrecision<DefaultPolicy>>());
        normwise("erf best", x, y, got, (er, ei), TOL_BEST);
    }
}

#[test]
fn erfc_at_best_precision() {
    for &(x, y, _, _, cr, ci) in ERF_REFS.iter() {
        let got = parts(c(x, y).erfc_p::<BestPrecision<DefaultPolicy>>());
        normwise("erfc best", x, y, got, (cr, ci), TOL_BEST);
    }
}

/// The lower tiers are allowed to be looser but not wrong: no argument may fall off a
/// cliff.
#[test]
fn erf_at_performance_precision() {
    for &(x, y, er, ei, _, _) in ERF_REFS.iter() {
        normwise(
            "erf perf",
            x,
            y,
            parts(c(x, y).erf_p::<Performance>()),
            (er, ei),
            TOL_DEFAULT,
        );
    }
}

/// The ray that exposed it, swept finely straight through the old failure band. All of
/// these previously took the series at a cancellation of `e^{2y^2}`. Past `|z| = 6.4`
/// that is more than f64 has.
#[test]
fn erf_on_the_45_degree_rays() {
    let mut a = 0.25f64;
    while a <= 9.0 {
        for sign in [-1.0f64, 1.0] {
            let (x, y) = (a, sign * a);
            // erf(conj z) = conj(erf z), so one mpmath value covers both rays.
            let want = ERF_REFS
                .iter()
                .find(|r| (r.0 - x).abs() < 1e-12 && (r.1 - y).abs() < 1e-12);
            if let Some(&(_, _, er, ei, _, _)) = want {
                let got = parts(c(x, y).erf_p::<BestPrecision<DefaultPolicy>>());
                normwise("erf ray", x, y, got, (er, ei), TOL_BEST);
            }
            // Independent of any table: |erf| is bounded on these rays, so a blow-up
            // is caught even where no reference row exists.
            let (gr, gi) = parts(c(x, y).erf());
            let norm = (gr * gr + gi * gi).sqrt();
            assert!(
                norm < 2.0,
                "erf({x} + {y}i) = ({gr:e}, {gi:e}), norm {norm:e} - |erf| <= 1.2 on this ray"
            );
        }
        a += 0.25;
    }
}

/// `erf` is odd and conjugate-symmetric. Both must survive the regime selection, which
/// now keys on `Im(z)^2` and so is symmetric in `Im z` by construction.
///
/// Graded normwise rather than per component, and deliberately so. On the imaginary
/// axis `erf(iy)` is purely imaginary, which the _series_ delivers structurally (every
/// term is purely imaginary) and `w` does not: where `w` is selected it leaves a real
/// part of a few times 1e-9 against an `|erf|` in the tens, i.e. ~2e-10 normwise, which
/// is inside `w`'s own accuracy.
///
/// That exactness was never a guarantee of `erf`, only of one of its two regimes, and
/// has always stopped at whatever `y` the regime boundary sits at: `y = 2.83` before
/// this fix and still `y = 2.83` at the default tier, `y = 2` at `Best`, where `w` is
/// accurate enough to be worth taking earlier. Asserting it per component would be
/// asserting something the implementation does not claim.
#[test]
fn erf_symmetries_hold_across_the_regime_boundary() {
    for &(x, y, _, _, _, _) in ERF_REFS.iter() {
        let p = parts(c(x, y).erf());
        let conj = parts(c(x, -y).erf());
        normwise("conjugate symmetry", x, y, (conj.0, -conj.1), p, TOL_DEFAULT);

        let odd = parts(c(-x, -y).erf());
        normwise("oddness", x, y, (-odd.0, -odd.1), p, TOL_DEFAULT);
    }
}
