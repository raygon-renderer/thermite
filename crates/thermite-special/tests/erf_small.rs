//! `erf` at small arguments, where `1 - m e^{-x^2}` has only absolute accuracy.
//!
//! At `Best` and above the f64 kernel takes fdlibm's `x + x R(x^2)/S(x^2)` below 0.84375
//! and is relatively accurate all the way down. The f32 kernel has had a small-argument arm
//! from `Average`. The default f64 tier keeps the cheap form and its absolute error, which
//! is pinned here too so a change in either direction is noticed.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]
#![allow(clippy::excessive_precision)]

use thermite::math::policy::DefaultPolicy;
use thermite::math::policy::policies::BestPrecision;
use thermite::prelude::*;
use thermite_special::{SpecialMath, SpecialMathWithPolicy};

include!("common/wide.rs");

type D = Vector<f64>;
type F = Vector<f32>;
type Best = BestPrecision<DefaultPolicy>;

/// `(x, erf(x))` from mpmath at 50 digits at the exact binary `x`, straddling the 0.84375
/// seam.
const REFS: &[(f64, f64)] = &[
    (1e-300, 1.1283791670955126e-300),
    (1e-20, 1.1283791670955125e-20),
    (1e-10, 1.1283791670955126e-10),
    (1e-05, 1.1283791670579e-05),
    (0.001, 0.0011283787909692365),
    (0.01, 0.011283415555849618),
    (0.05, 0.05637197779701663),
    (0.1, 0.1124629160182849),
    (0.25, 0.27632639016823696),
    (0.4, 0.42839235504666845),
    (0.5, 0.5204998778130465),
    (0.7, 0.6778011938374184),
    (0.8, 0.7421009647076605),
    (0.84, 0.7651427114549945),
    (0.84375, 0.7672256612323416),
    (0.85, 0.7706680576083526),
    (1.0, 0.8427007929497149),
    (2.0, 0.9953222650189527),
];

fn rel_ulps(got: f64, want: f64, eps: f64) -> f64 {
    ((got - want) / want).abs() / eps
}

#[test]
fn f64_best_is_relatively_accurate_down_to_the_denormals() {
    let mut worst = (0.0, 0.0);
    for &(x, want) in REFS {
        for (x, want) in [(x, want), (-x, -want)] {
            let got = D::splat(x).erf_p::<Best>().extract::<0>();
            let u = rel_ulps(got, want, f64::EPSILON);
            assert!(u <= 2.0, "erf best({x:e}): got {got:e}, want {want:e}, {u:.2} ulp");
            if u > worst.0 {
                worst = (u, x);
            }
        }
    }
    eprintln!("f64 best worst {:.2} ulp at x = {:e}", worst.0, worst.1);

    // Exact at zero, with the sign of the zero.
    assert_eq!(D::splat(0.0).erf_p::<Best>().extract::<0>().to_bits(), 0.0f64.to_bits());
    assert_eq!(
        D::splat(-0.0).erf_p::<Best>().extract::<0>().to_bits(),
        (-0.0f64).to_bits()
    );
}

/// The default tier is unchanged: absolute error of about an ulp of 1, so `erf(0)` is not
/// zero and `erf(1e-8)` is 2e-8 relative. Pinned so a regression in either tier is visible.
#[test]
fn f64_default_keeps_the_cheap_form() {
    for &(x, want) in REFS {
        let got = D::splat(x).erf().extract::<0>();
        assert!(
            (got - want).abs() <= 4.0 * f64::EPSILON,
            "erf default({x:e}): got {got:e}, want {want:e}"
        );
    }
    let at_zero = D::splat(0.0).erf().extract::<0>().abs();
    assert!(
        at_zero > 0.0 && at_zero <= 4.0 * f64::EPSILON,
        "erf default(0) = {at_zero:e}"
    );
}

/// The f32 default tier's arm is a 7-term Taylor series used up to `x = 1`, whose truncation
/// is the dominant error near the top of that range (13 ulp at 0.8). `Best` takes the Pade.
#[test]
fn f32_has_had_the_arm_from_average() {
    let mut worst_def = (0.0, 0.0);
    for &(x, want) in REFS {
        let x = x as f32;
        if x == 0.0 {
            continue;
        }
        let want = want as f32 as f64;
        let def = F::splat(x).erf().extract::<0>() as f64;
        let best = F::splat(x).erf_p::<Best>().extract::<0>() as f64;
        let u = rel_ulps(def, want, f32::EPSILON as f64);
        assert!(u <= 32.0, "erff default({x:e}): {def:e} vs {want:e}, {u:.1} ulp");
        if u > worst_def.0 {
            worst_def = (u, x);
        }
        assert!(
            rel_ulps(best, want, f32::EPSILON as f64) <= 3.0,
            "erff best({x:e}): {best:e} vs {want:e}"
        );
    }
    eprintln!("f32 default worst {:.2} ulp at x = {:e}", worst_def.0, worst_def.1);
    assert_eq!(F::splat(0.0).erf_p::<Best>().extract::<0>(), 0.0);
}

/// Both arms in one packet, each lane bit-identical to a splat of itself on the same
/// backend.
#[test]
fn packet_mixes_both_arms_bit_exactly() {
    let xs = [1e-9f64, -0.3, 0.84, -3.0];
    let got = f64x4::new(xs).erf_p::<Best>().into_array();
    for (k, &x) in xs.iter().enumerate() {
        assert_eq!(
            got[k].to_bits(),
            f64x4::splat(x).erf_p::<Best>().extract::<0>().to_bits(),
            "lane {k}"
        );
    }
}
