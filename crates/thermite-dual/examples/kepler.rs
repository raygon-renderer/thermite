//! Solving Kepler's equation by Newton's method, without ever writing the derivative.
//!
//! `M = E - e sin(E)` relates the *mean anomaly* `M` (which advances linearly with
//! time) to the *eccentric anomaly* `E` (which gives the actual position on the
//! orbit). It has no closed-form inverse, so every orbit propagator solves it
//! numerically, and Newton's method is the standard choice.
//!
//! Newton needs both `f(E)` and `f'(E)`. Here only `f` is ever written:
//!
//! ```text
//! #[inline(always)]
//! fn residual<W: FloatVector + TranscendentalMath>(e: W, ecc: W, m: W) -> W {
//!     ecc.nmul_adde(e.sin(), e - m)     // e - ecc*sin(e) - m, fused
//! }
//! ```
//!
//! `residual` is an ordinary Thermite kernel. It names no backend, no lane count,
//! and no element type, and it has no idea `Dual` exists. Evaluating it through
//! [`AutoDiff::ad`] instantiates it at `W = Dual<V, 1>`, and the result carries the
//! value in `.re` and the derivative in `.dual[0]` from **one** evaluation.
//!
//! Run with:
//!
//! ```text
//! cargo run --release -p thermite-dual --example kepler
//! ```
//!
//! # Why bother, when this derivative is easy?
//!
//! It is: `f'(E) = 1 - e cos(E)`, and the example asserts the dual part matches it
//! exactly. That check is the point. Verifying automatic differentiation against a
//! derivative you *can* write by hand is what earns the right to trust it on one you
//! cannot.
//!
//! The two things that do not scale are writing the derivative and keeping it in
//! sync. Change `residual` here and the Newton step stays correct for free; change
//! it with a hand-written `dresidual` beside it and nothing tells you the two have
//! drifted apart except a solver that quietly converges more slowly.
//!
//! The accuracy is not a wash either. The example prints a central finite
//! difference beside the dual part, using `cbrt(eps)` as the step, which is the
//! best a central difference can do: truncation error falls as the step shrinks
//! and cancellation error grows, and that is where they cross. It still lands
//! around `3e-11` here.
//!
//! The dual part's error is `0`. Not small, exactly zero, and the example asserts
//! it. Forward-mode AD does not approximate a derivative, it evaluates the
//! derivative's own arithmetic alongside the value.
//!
//! # Still SIMD
//!
//! `Dual<V, N>` delegates the whole vector-trait surface to its inner `V`, so all of
//! this runs a register of orbits at a time: each lane is a different sample time on
//! the same orbit. Lanes converge at different iterations, so a mask freezes the
//! ones that are done, exactly like the escape test in the `mandelbrot` example over
//! in core.

use thermite::math::TranscendentalMath;
use thermite::prelude::*;
use thermite::simd::{FloatSimd, SizedSimd};
use thermite_dual::{AutoDiff, Dual, DualValue};

/// Newton is quadratic, so this is a generous ceiling. Even `e = 0.99` converges in
/// single digits from this starting guess.
const MAX_ITERS: u32 = 20;

/// Kepler's equation as a root-finding problem: `f(E) = E - e sin(E) - M`.
///
/// A completely ordinary generic kernel. Nothing here is aware of derivatives.
///
/// Written as a negated fused multiply-add rather than the literal spelling:
/// `nmul_adde(a, b, c)` is `c - a * b`, so the multiply and the subtract issue as
/// one instruction with one rounding. The `e` suffix is the *estimating* form,
/// which uses the hardware FMA where there is one and does not pay for exact
/// emulation where there is not. That is the right trade here, since Newton
/// converges quadratically and re-derives its own accuracy every step.
#[inline(always)]
fn residual<W: FloatVector + TranscendentalMath>(ecc_anom: W, ecc: W, m: W) -> W {
    ecc.nmul_adde(ecc_anom.sin(), ecc_anom - m)
}

/// The hand-written derivative, used *only* to check the dual part.
#[inline(always)]
fn dresidual<W: FloatVector + TranscendentalMath>(ecc_anom: W, ecc: W) -> W {
    ecc.nmul_adde(ecc_anom.cos(), W::ONE)
}

/// Solves for the eccentric anomaly of every lane at once.
///
/// Returns the solution and how many iterations the slowest lane needed.
#[inline(always)]
fn solve<V>(m: V, ecc: V, tol: V) -> (V, u32)
where
    V: FloatVector<Element = f64> + TranscendentalMath + DualValue,
    Dual<V, 1>: FloatVector + TranscendentalMath,
{
    // Danby's starting guess, `E0 = M + 0.85 e sign(sin M)`.
    //
    // The obvious `E0 = M` is exact for a circular orbit and fine up to moderate
    // eccentricity, but it fails outright as `e` approaches 1: near the periapsis
    // the derivative `1 - e cos E` collapses toward `1 - e`, so the first Newton
    // step divides by something tiny and throws the iterate across the orbit.
    // Starting ahead of `M` by a fraction of `e` puts the guess on the far side of
    // that stiff region, where Newton is well behaved.
    //
    // Nothing about this is specific to SIMD. It is just what solving Kepler
    // correctly requires, and leaving it out is how the example would quietly work
    // for `e = 0.5` and diverge for `e = 0.99`.
    let mut e_anom = ecc.mul_adde(m.sin().signum() * V::splat(0.85), m);
    let mut used = 0;

    for i in 0..MAX_ITERS {
        // One evaluation, both quantities. `ecc` and `m` are lifted to constants so
        // only `E` is seeded as an independent variable, which is what makes
        // `.dual[0]` mean df/dE.
        let r = (|x| residual(x, Dual::constant(ecc), Dual::constant(m))).ad([e_anom]);

        let f = r.re;
        let df = r.dual[0];

        // Lanes still outside tolerance. Once none are left the remaining
        // iterations cannot change the answer.
        let active = f.abs().cmp_gt(tol);
        if !active.any() {
            break;
        }

        // `f'(E) = 1 - e cos(E)` and `e < 1`, so the derivative is bounded away from
        // zero on every physical orbit and this division is always safe.
        //
        // `sub_c` updates only the lanes still converging, leaving the finished ones
        // bit-for-bit alone. The mask is the first argument, as it is for every
        // masked variant.
        e_anom = e_anom.sub_c(active, f / df);
        used = i + 1;
    }

    (e_anom, used)
}

/// The whole demo, dispatched once so the body gets per-ISA codegen.
#[thermite::dispatch(S)]
pub fn demo<S: FloatSimd<f64>>() {
    println!("dispatched to {:?}\n", S::ISA);

    type V<S> = Vector<<S as SizedSimd<f64, i64, u64>>::fxN>;

    let lanes = V::<S>::LANES;
    let tol = V::<S>::splat(1e-14);

    // One register of sample times: mean anomaly swept across most of an orbit.
    // Exactly how a propagator batches work, and the same shape as the coordinate
    // ramp in the mandelbrot example.
    let m = V::<S>::indexed() * V::<S>::splat(2.8 / lanes as f64) + V::<S>::splat(0.15);

    println!("solving {lanes} sample times per register\n");
    println!("  ecc    iters    max |residual|    max |dual - hand|      FD error");
    println!("  ---------------------------------------------------------------------");

    for &e in &[0.0f64, 0.2, 0.5, 0.9, 0.99] {
        let ecc = V::<S>::splat(e);
        let (e_anom, iters) = solve(m, ecc, tol);

        // Check 1: substitute back. This is the only check that proves the answer is
        // right, and it needs no reference data.
        let residual_err = residual(e_anom, ecc, m).abs().max_element();

        // Check 2: the dual part against the derivative written by hand.
        let r = (|x| residual(x, Dual::constant(ecc), Dual::constant(m))).ad([e_anom]);
        let hand = dresidual(e_anom, ecc);
        let dual_err = (r.dual[0] - hand).abs().max_element();

        // Check 3: what a central finite difference would have cost. `cbrt(eps)` is
        // the textbook optimal step for a central difference, and it still leaves
        // about 3e-11 of error here against a derivative of order 1.
        let h = V::<S>::splat(f64::EPSILON.cbrt());
        let fd = (residual(e_anom + h, ecc, m) - residual(e_anom - h, ecc, m)) / (h + h);
        let fd_err = (fd - hand).abs().max_element();

        println!("  {e:<6} {iters:>3}      {residual_err:>10.3e}       {dual_err:>10.3e}        {fd_err:>10.3e}");

        assert!(residual_err < 1e-13, "Newton did not converge for e = {e}");
        assert_eq!(dual_err, 0.0, "the dual part should equal the hand derivative exactly");
    }

    // The bonus: seeding all three inputs turns the same single pass into a full
    // gradient. `.dual[0]` is what Newton used; the other two are the sensitivity of
    // the residual to eccentricity and to mean anomaly, which came along for free.
    let ecc = V::<S>::splat(0.5);
    let (e_anom, _) = solve(m, ecc, tol);
    let g = residual.ad([e_anom, ecc, m]);

    // d/de (E - e sin E - M) = -sin E, and d/dM = -1.
    assert_eq!((g.dual[1] + e_anom.sin()).abs().max_element(), 0.0);
    assert_eq!((g.dual[2] + V::<S>::ONE).abs().max_element(), 0.0);

    println!("\ngradient from the same pass: df/dE, df/de = -sin(E), df/dM = -1");
}

fn main() {
    thermite::dispatch_dyn!(demo());
}
