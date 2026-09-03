//! The Fresnel integrals `C(x)` and `S(x)`.
//!
//! References are mpmath at 60 digits, evaluated at the exact binary value in the
//! first column rather than at the decimal that produced it. `REFS_F32` is a
//! separate table anchored at f32 arguments, because `dC/dx` is O(1) and reading the
//! f64 rows at f32 would charge the kernel `x` ulps for its own input rounding.
//!
//! The rows past `x = 1e4` are the point of this file. `C` and `S` are `1/2` plus a
//! term of size `1/(pi x)` there, so the whole accuracy question is whether the phase
//! `pi x^2/2` survives. Computed as `x*x*0.5` it is 5.3e-6 wrong at `x = 98765` and
//! has the wrong sign by `x ~ 1e9`.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]
#![allow(clippy::excessive_precision)]

use thermite::Vector;
use thermite::math::policy::DefaultPolicy;
use thermite::math::policy::policies::{AvoidBranching, BestPrecision, WorstPrecision};
use thermite::prelude::*;
use thermite_special::{RealSpecialMath, RealSpecialMathWithPolicy};

type D = Vector<f64>;
type F = Vector<f32>;

include!("fresnel_ref/table.rs");

const EPS64: f64 = f64::EPSILON;
const EPS32: f64 = f32::EPSILON as f64;

/// Error in ulps of `want`. `C` and `S` have no zeros on `x > 0`, so plain relative
/// accuracy is the honest contract. Only the exact zero at the origin needs an
/// absolute arm.
#[track_caller]
fn close(name: &str, x: f64, got: f64, want: f64, ulps: f64, eps: f64) {
    let err = if want == 0.0 {
        (got - want).abs() / eps
    } else {
        ((got - want) / want).abs() / eps
    };
    assert!(
        err <= ulps,
        "{name}({x:e}): got {got:e}, want {want:e} ({err:.2} ulp, limit {ulps})"
    );
}

/// Measured worst over these rows is 2.8 ulp (f64) and 3.4 (f32). The limits are set
/// well above that so ordinary retuning does not trip them, and far below the
/// thousands a broken phase or a monomial-basis small branch would produce.
const TOL64: f64 = 16.0;
const TOL32: f64 = 24.0;

#[test]
fn fresnel_f64_table() {
    for &(x, c, s) in REFS.iter() {
        let (gc, gs) = D::splat(x).fresnel();
        close("C", x, gc.extract::<0>(), c, TOL64, EPS64);
        close("S", x, gs.extract::<0>(), s, TOL64, EPS64);
    }
}

#[test]
fn fresnel_f64_best() {
    for &(x, c, s) in REFS.iter() {
        let (gc, gs) = D::splat(x).fresnel_p::<BestPrecision<DefaultPolicy>>();
        close("C best", x, gc.extract::<0>(), c, TOL64, EPS64);
        close("S best", x, gs.extract::<0>(), s, TOL64, EPS64);
    }
}

/// `S(x) ~ pi x^3/6`, so it underflows f32 for `x` below about 1e-14 while the f64
/// reference still carries a value. Zero is the right answer there, and saying so is
/// worth more than skipping the row.
fn underflows_f32(want: f64) -> bool {
    want != 0.0 && (want as f32) == 0.0
}

#[test]
fn fresnel_f32_table() {
    for &(x, c, s) in REFS_F32.iter() {
        let (gc, gs) = F::splat(x as f32).fresnel();
        let (gc, gs) = (gc.extract::<0>() as f64, gs.extract::<0>() as f64);
        for (name, got, want) in [("C f32", gc, c), ("S f32", gs, s)] {
            if underflows_f32(want) {
                assert_eq!(got, 0.0, "{name}({x:e}): underflows f32, expected 0");
            } else {
                close(name, x, got, want, TOL32, EPS32);
            }
        }
    }
}

/// The regression test for the two-word phase reduction. These are the arguments
/// where `sin(pi*(x*x*0.5))` measures 5.2e-13, 9.8e-11, 5.3e-6 and a wrong sign. If
/// the reduction is ever simplified back to `x*x*0.5`, this fails by orders of
/// magnitude and nothing else in the file notices.
#[test]
fn fresnel_phase_is_exact_at_large_arguments() {
    let mut seen = 0;
    for &(x, c, s) in REFS.iter().filter(|r| r.0 >= 100.0 && r.0 <= 1e15) {
        let (gc, gs) = D::splat(x).fresnel();
        close("C phase", x, gc.extract::<0>(), c, TOL64, EPS64);
        close("S phase", x, gs.extract::<0>(), s, TOL64, EPS64);
        seen += 1;
    }
    assert!(seen >= 8, "the large-argument rows went missing from the table");
}

/// Below `Average` the residual is dropped by design, so the phase is only as good as
/// `x*x`. The kernel must still be sane everywhere and accurate where the phase is
/// not yet in question.
#[test]
fn fresnel_low_tier_stays_bounded() {
    for &(x, _, _) in REFS.iter() {
        let (gc, gs) = D::splat(x).fresnel_p::<WorstPrecision<DefaultPolicy>>();
        let (c, s) = (gc.extract::<0>(), gs.extract::<0>());
        assert!(c.is_finite() && s.is_finite(), "x={x:e}: {c} {s}");
        assert!(
            (-0.01..=1.0).contains(&c) && (-0.01..=1.0).contains(&s),
            "x={x:e}: {c} {s}"
        );
    }
    let &(x, c, s) = REFS.iter().find(|r| r.0 == 1.5).expect("x = 1.5 row");
    let (gc, gs) = D::splat(x).fresnel_p::<WorstPrecision<DefaultPolicy>>();
    close("C worst", x, gc.extract::<0>(), c, 4096.0, EPS64);
    close("S worst", x, gs.extract::<0>(), s, 4096.0, EPS64);
}

#[test]
fn fresnel_is_odd() {
    for &(x, c, s) in REFS.iter() {
        let (gc, gs) = D::splat(-x).fresnel();
        close("C(-x)", x, gc.extract::<0>(), -c, TOL64, EPS64);
        close("S(-x)", x, gs.extract::<0>(), -s, TOL64, EPS64);
    }
}

#[test]
fn fresnel_singles_match_the_pair() {
    for &(x, _, _) in REFS.iter() {
        let v = D::splat(x);
        let (c, s) = v.fresnel();
        assert_eq!(v.fresnel_c().extract::<0>(), c.extract::<0>(), "C at {x:e}");
        assert_eq!(v.fresnel_s().extract::<0>(), s.extract::<0>(), "S at {x:e}");
    }
}

#[test]
fn fresnel_edges() {
    let (c, s) = D::splat(0.0).fresnel();
    assert_eq!(c.extract::<0>(), 0.0);
    assert_eq!(s.extract::<0>(), 0.0);

    // Odd, so -0.0 comes back as -0.0 rather than +0.0.
    let (c, s) = D::splat(-0.0).fresnel();
    assert!(c.extract::<0>() == 0.0 && c.extract::<0>().is_sign_negative());
    assert!(s.extract::<0>() == 0.0 && s.extract::<0>().is_sign_negative());

    let (c, s) = D::splat(f64::INFINITY).fresnel();
    assert_eq!(c.extract::<0>(), 0.5);
    assert_eq!(s.extract::<0>(), 0.5);

    let (c, s) = D::splat(f64::NEG_INFINITY).fresnel();
    assert_eq!(c.extract::<0>(), -0.5);
    assert_eq!(s.extract::<0>(), -0.5);

    let (c, s) = D::splat(f64::NAN).fresnel();
    assert!(c.extract::<0>().is_nan() && s.extract::<0>().is_nan());
}

/// The `is_small.all()` fast path is a per-vector decision, so a vector straddling
/// the crossover takes a different road through the kernel than a uniform one. They
/// must agree bit for bit.
#[test]
fn fresnel_mixed_lanes_agree_with_uniform() {
    const LANES: usize = <D as GenericVector>::LANES;
    let xs = [0.5, 3.0, 1.0, 1e6, 2.5265, 300.0, 0.001, 12345.678];
    let mut buf = [0.0f64; LANES];
    for k in 0..LANES {
        buf[k] = xs[k % xs.len()];
    }
    let (mc, ms) = D::new(buf).fresnel();
    let (mc, ms) = (mc.into_array(), ms.into_array());
    for k in 0..LANES {
        let (uc, us) = D::splat(buf[k]).fresnel();
        assert_eq!(mc[k], uc.extract::<0>(), "C lane {k} (x={})", buf[k]);
        assert_eq!(ms[k], us.extract::<0>(), "S lane {k} (x={})", buf[k]);
    }
}

/// With `avoid_branching` both arms are computed unconditionally, including where the
/// dead one overflows to infinity or produces a NaN. The select must still land.
#[test]
fn fresnel_branchless_matches() {
    for &(x, c, s) in REFS.iter() {
        let (gc, gs) = D::splat(x).fresnel_p::<AvoidBranching<DefaultPolicy, true>>();
        close("C nobranch", x, gc.extract::<0>(), c, TOL64, EPS64);
        close("S nobranch", x, gs.extract::<0>(), s, TOL64, EPS64);
    }
}
