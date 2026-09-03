//! The trigonometric integrals `Si(x)` and `Ci(x)`.
//!
//! References are mpmath at 60 digits at the exact binary argument, with a separate
//! f32-anchored table (see `fresnel.rs` for why).
//!
//! `Ci` is graded against the fourth table column rather than against its own value,
//! for two independent reasons. It has zeros (the first at `x ~ 0.6165`, which is in
//! the table) and nothing is relatively accurate at one. And below the crossover it
//! is `(gamma + ln x) - Cin`, a difference of two terms that grow like `ln x` while
//! the result decays like `1/x`, so the cancellation reaches 50x by `x = 12`. The
//! column is the size of what the kernel actually combines: the two terms below the
//! crossover, and the `1/x` oscillation above it. That crossover differs by precision,
//! so `REFS` and `REFS_F32` carry different envelopes for the same `x`.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]
#![allow(clippy::excessive_precision)]

use thermite::Vector;
use thermite::math::policy::DefaultPolicy;
use thermite::math::policy::policies::{AvoidBranching, BestPrecision};
use thermite::prelude::*;
use thermite_special::{RealSpecialMath, RealSpecialMathWithPolicy};

type D = Vector<f64>;
type F = Vector<f32>;

include!("sici_ref/table.rs");

const EPS64: f64 = f64::EPSILON;
const EPS32: f64 = f32::EPSILON as f64;

/// Measured worst is 2.03 ulp (`Si`) and 1.42 (`Ci`, on its envelope) in f64, 1.34 and
/// 1.99 in f32.
const TOL64: f64 = 16.0;
const TOL32: f64 = 24.0;

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

/// `Ci` graded against the fourth table column: the scale the kernel's arithmetic can
/// actually deliver at that `x`, which below the crossover is the pair of terms it
/// subtracts and above it the `1/x` oscillation. The cancellation this admits is real
/// and not a coefficient problem: at `x = 11.9` the two terms are both about 2.7 while
/// `Ci` is -0.057, so 50x of the relative accuracy is gone before the fit is consulted.
#[track_caller]
fn close_env(name: &str, x: f64, got: f64, want: f64, env: f64, ulps: f64, eps: f64) {
    let err = (got - want).abs() / env / eps;
    assert!(
        err <= ulps,
        "{name}({x:e}): got {got:e}, want {want:e} ({err:.2} envelope-ulp, limit {ulps})"
    );
}

/// Where the DEFAULT tier's `sin_cos` still determines `Ci`'s phase. `Ci`'s value is
/// its oscillation, so beyond this the answer is whatever the argument reduction left,
/// and full reduction is `Best`+. Measured: f64 is fine at `x = 9.9e8` and gone by
/// `7.7e14`. f32 is fine at `1e4` and gone by `9.9e4`.
/// `sici_ci_large_argument_needs_best` covers everything past these at `Best`.
const CI_DEFAULT_TIER_LIMIT_F64: f64 = 1e9;
const CI_DEFAULT_TIER_LIMIT_F32: f64 = 1e4;

#[test]
fn sici_f64_table() {
    for &(x, si, ci, env) in REFS.iter() {
        if x == 0.0 {
            continue; // Ci(0) = -inf; covered in sici_edges
        }
        let (gsi, gci) = D::splat(x).sici();
        close("Si", x, gsi.extract::<0>(), si, TOL64, EPS64);
        if x <= CI_DEFAULT_TIER_LIMIT_F64 {
            close_env("Ci", x, gci.extract::<0>(), ci, env, TOL64, EPS64);
        }
    }
}

#[test]
fn sici_f64_best() {
    for &(x, si, ci, env) in REFS.iter() {
        if x == 0.0 {
            continue;
        }
        let (gsi, gci) = D::splat(x).sici_p::<BestPrecision<DefaultPolicy>>();
        close("Si best", x, gsi.extract::<0>(), si, TOL64, EPS64);
        close_env("Ci best", x, gci.extract::<0>(), ci, env, TOL64, EPS64);
    }
}

#[test]
fn sici_f32_table() {
    for &(x, si, ci, env) in REFS_F32.iter() {
        if x == 0.0 {
            continue;
        }
        let (gsi, gci) = F::splat(x as f32).sici();
        close("Si f32", x, gsi.extract::<0>() as f64, si, TOL32, EPS32);
        // `Ci`'s value IS the oscillation, so its accuracy is `sin_cos`'s argument
        // reduction, and full reduction is a Best-tier property. See
        // `sici_ci_large_argument_needs_best`, which pins where that starts to bite.
        if x <= CI_DEFAULT_TIER_LIMIT_F32 {
            close_env("Ci f32", x, gci.extract::<0>() as f64, ci, env, TOL32, EPS32);
        }
    }
}

/// The tier inheritance in the docs, measured rather than asserted.
///
/// `Ci = f sin x - g cos x`, so a phase error is a relative error, and thermite's
/// full argument reduction is `Best`+. At the default tier f32 `Ci(98765.4)` comes
/// back as -1.03e-10 where the answer is 4.03e-7, not close, and `Best` returns
/// 4.0321078e-7. f64 holds the default tier much further out but goes the same way by
/// `x = 1e15` (-1e-30 against 8.58e-16).
///
/// `Si` is insulated: it tends to pi/2 with the oscillation only a `1/x` correction,
/// so the same phase error is invisible in it.
#[test]
fn sici_ci_large_argument_needs_best() {
    for &(x, _, ci, env) in REFS.iter().filter(|r| r.0 >= 1e4 && r.0 <= 1e15) {
        let got = D::splat(x).sici_p::<BestPrecision<DefaultPolicy>>().1.extract::<0>();
        close_env("Ci best", x, got, ci, env, TOL64, EPS64);
    }
    for &(x, _, ci, env) in REFS_F32.iter().filter(|r| r.0 >= 1e4 && r.0 <= 2.1e7) {
        let got = F::splat(x as f32)
            .sici_p::<BestPrecision<DefaultPolicy>>()
            .1
            .extract::<0>() as f64;
        close_env("Ci f32 best", x, got, ci, env, TOL32, EPS32);
    }
}

/// `Ci`'s first zero. Relative accuracy is impossible. The contract is that the
/// absolute error stays at the level of the terms being cancelled, which are O(1).
#[test]
fn sici_ci_at_its_zero() {
    let &(x, _, ci, _) = REFS
        .iter()
        .find(|r| (r.0 - 0.6165054856207162).abs() < 1e-12)
        .expect("the zero row");
    let got = D::splat(x).sici().1.extract::<0>();
    assert!(ci.abs() < 1e-16, "the table row is not actually the zero: {ci:e}");
    assert!(
        (got - ci).abs() < 16.0 * EPS64,
        "Ci at its zero: got {got:e}, want {ci:e} (absolute {:e})",
        (got - ci).abs()
    );
}

#[test]
fn sici_sign_conventions() {
    // Si is odd. Ci(-x) = Ci(|x|), dropping the imaginary i*pi. Both match SciPy.
    for &(x, si, ci, env) in REFS.iter() {
        if x == 0.0 {
            continue;
        }
        let (gsi, gci) = D::splat(-x).sici();
        close("Si(-x)", x, gsi.extract::<0>(), -si, TOL64, EPS64);
        if x <= CI_DEFAULT_TIER_LIMIT_F64 {
            close_env("Ci(-x)", x, gci.extract::<0>(), ci, env, TOL64, EPS64);
        }
    }
}

#[test]
fn sici_singles_match_the_pair() {
    for &(x, _, _, _) in REFS.iter() {
        let v = D::splat(x);
        let (si, ci) = v.sici();
        assert_eq!(v.sinint().extract::<0>(), si.extract::<0>(), "Si at {x:e}");
        assert_eq!(v.cosint().extract::<0>(), ci.extract::<0>(), "Ci at {x:e}");
    }
}

#[test]
fn sici_edges() {
    let (si, ci) = D::splat(0.0).sici();
    assert_eq!(si.extract::<0>(), 0.0);
    assert_eq!(ci.extract::<0>(), f64::NEG_INFINITY);

    let (si, ci) = D::splat(-0.0).sici();
    assert!(si.extract::<0>() == 0.0 && si.extract::<0>().is_sign_negative());
    assert_eq!(ci.extract::<0>(), f64::NEG_INFINITY);

    let (si, ci) = D::splat(f64::INFINITY).sici();
    assert_eq!(si.extract::<0>(), core::f64::consts::FRAC_PI_2);
    assert_eq!(ci.extract::<0>(), 0.0);

    let (si, ci) = D::splat(f64::NEG_INFINITY).sici();
    assert_eq!(si.extract::<0>(), -core::f64::consts::FRAC_PI_2);
    assert_eq!(ci.extract::<0>(), 0.0);

    let (si, ci) = D::splat(f64::NAN).sici();
    assert!(si.extract::<0>().is_nan() && ci.extract::<0>().is_nan());
}

#[test]
fn sici_mixed_lanes_agree_with_uniform() {
    const LANES: usize = <D as GenericVector>::LANES;
    let xs = [0.5, 30.0, 1.0, 1e6, 12.0, 3.0, 0.001, 12345.678];
    let mut buf = [0.0f64; LANES];
    for k in 0..LANES {
        buf[k] = xs[k % xs.len()];
    }
    let (msi, mci) = D::new(buf).sici();
    let (msi, mci) = (msi.into_array(), mci.into_array());
    for k in 0..LANES {
        let (usi, uci) = D::splat(buf[k]).sici();
        assert_eq!(msi[k], usi.extract::<0>(), "Si lane {k} (x={})", buf[k]);
        assert_eq!(mci[k], uci.extract::<0>(), "Ci lane {k} (x={})", buf[k]);
    }
}

#[test]
fn sici_branchless_matches() {
    for &(x, si, ci, env) in REFS.iter() {
        if x == 0.0 {
            continue;
        }
        let (gsi, gci) = D::splat(x).sici_p::<AvoidBranching<DefaultPolicy, true>>();
        close("Si nobranch", x, gsi.extract::<0>(), si, TOL64, EPS64);
        if x <= CI_DEFAULT_TIER_LIMIT_F64 {
            close_env("Ci nobranch", x, gci.extract::<0>(), ci, env, TOL64, EPS64);
        }
    }
}
