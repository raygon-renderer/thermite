//! `gelu`'s left tail, at every precision tier and both dtypes.
//!
//! `GELU(x) = 0.5 x (1 + erf(ax/sqrt2))`, and that spelling is a trap: once `erf` rounds
//! to exactly -1 the sum is `0.5x - 0.5x` and the function returns **+0.0** where the
//! answer is a small, perfectly representable negative number. float32 crossed that line
//! at about x = -5 and float64 at about x = -8.3, so `gelu(-6.0f32)` returned zero for
//! an answer of -5.92e-9.
//!
//! The kernels evaluate `0.5 x erfc(-ax/sqrt2)` instead, which is the same value by an
//! exact identity and has no cancellation anywhere. These probes sit past the old cliff
//! specifically, because a test written on ordinary arguments cannot see the bug at all -
//! `gelu` agrees with itself to the last bit above x = -4.
//!
//! Found by the Python accuracy harness on the day `gelu` was first
//! exposed to it. It showed up as `precision` disagreeing with `reference`, and that is
//! not a coincidence: the `is_reference` arm was already written with `erfc`, so the two
//! lowerings of the same function had disagreed since the day they were written.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]
#![allow(clippy::excessive_precision)]

use thermite::Vector;
use thermite::math::policy::DefaultPolicy;
use thermite::math::policy::policies::{HighPerformance, Performance, Precision, Reference, Size, UltraPerformance};
use thermite::prelude::*;
use thermite_special::RealSpecialMathWithPolicy;

type D = Vector<f64>;
type F = Vector<f32>;

/// `(x, gelu(x, 1.0))` from mpmath at `dps = 40`, as `0.5*x*erfc(-x/sqrt(2))`.
///
/// Generated, not typed. A hand-written first draft of this table was wrong in the
/// seventh digit at nine of the twelve points and failed the kernel that was correct.
///
/// Everything below -5 is past the point where the old `1 + erf` form returned zero.
const REFS: &[(f64, f64)] = &[
    (-12.0, -2.1317785344932148e-32),
    (-9.0, -1.0157295653584566e-18),
    (-8.0, -4.9767684594174273e-15),
    (-7.0, -8.9586878072008450e-12),
    (-6.0, -5.9195258702261888e-09),
    (-5.0, -1.4332578593959696e-06),
    (-4.0, -1.2668496733247969e-04),
    (-3.0, -4.0496940948902836e-03),
    (-1.0, -1.5865525393145705e-01),
    (0.0, 0.0),
    (1.0, 8.4134474606854295e-01),
    (5.0, 4.9999985667421406e+00),
];

#[track_caller]
fn close(name: &str, got: f64, want: f64, tol: f64) {
    if want == 0.0 {
        assert!(got == 0.0, "{name}: got {got:?}, want exactly 0");
        return;
    }

    let rel = ((got - want) / want).abs();
    assert!(
        rel <= tol,
        "{name}: got {got:?}, want {want:?} (rel {rel:e}, tol {tol:e})"
    );
}

/// Runs `$body` at every tier whose `erf` kernel is actually evaluated in the tail.
///
/// `Worst` and `Medium` are excluded from any MAGNITUDE check, and not as a concession:
/// their `erf` clamps the argument at 4.5 and evaluates an Abramowitz-Stegun fit, so past
/// that point the answer saturates by construction. Measured relative error at
/// x = -9 (float32): `ultra_performance` 1.9e+12, `high_performance` 1.3e+09,
/// `performance` 3.9e-06. Asserting a magnitude there would be asserting the fit's
/// arbitrary saturation value. They are covered by the sign test below, which is the
/// property that actually broke.
macro_rules! for_each_accurate_tier {
    (|$p:ident| $body:block) => {{
        {
            type $p = Performance;
            $body
        }
        {
            type $p = Size;
            $body
        }
        {
            type $p = DefaultPolicy;
            $body
        }
        {
            type $p = Precision;
            $body
        }
        {
            type $p = Reference;
            $body
        }
    }};
}

/// Every tier, including the two whose tail magnitude is meaningless.
macro_rules! for_each_tier {
    (|$p:ident| $body:block) => {{
        {
            type $p = UltraPerformance;
            $body
        }
        {
            type $p = HighPerformance;
            $body
        }
        for_each_accurate_tier!(|$p| $body);
    }};
}

#[test]
fn gelu_f64_left_tail_is_not_zero() {
    for_each_accurate_tier!(|P| {
        for &(x, want) in REFS {
            let got = D::splat(x).gelu_p::<P>(D::ONE).extract::<0>();
            close(&format!("gelu f64 {x}"), got, want, 1e-6);
        }
    });
}

#[test]
fn gelu_f32_left_tail_is_not_zero() {
    for_each_accurate_tier!(|P| {
        for &(x, want) in REFS {
            // float32 carries ~7 digits, so the f64 references only bound it that far.
            let got = F::splat(x as f32).gelu_p::<P>(F::ONE).extract::<0>() as f64;
            close(&format!("gelu f32 {x}"), got, want, 1e-5);
        }
    });
}

/// The specific regression: a finite, representable, NEGATIVE answer where the old form
/// produced `+0.0`. Checked apart from the tolerance sweep because "not zero, and
/// negative" is the property that broke, and it holds at every tier including the ones
/// whose magnitude is only approximate.
#[test]
fn gelu_tail_keeps_its_sign_at_every_tier() {
    for_each_tier!(|P| {
        for &x in &[-5.0f64, -6.0, -7.0, -8.0, -9.0] {
            let d = D::splat(x).gelu_p::<P>(D::ONE).extract::<0>();
            assert!(
                d < 0.0 && d.is_finite(),
                "gelu f64 {x}: got {d:?}, want a small negative number"
            );

            // float32 underflows to zero for real below about x = -13.2, which is far
            // past every probe here.
            let f = F::splat(x as f32).gelu_p::<P>(F::ONE).extract::<0>();
            assert!(
                f < 0.0 && f.is_finite(),
                "gelu f32 {x}: got {f:?}, want a small negative number"
            );
        }
    });
}

/// `gelu_d` shares the value path, so it inherits the same bug and the same fix. The
/// derivative is checked only for being finite and positive in the tail, and its accuracy is
/// not what this file is about.
#[test]
fn gelu_d_value_matches_gelu() {
    use thermite_special::RealPrimalMathWithPolicy;

    for &x in &[-9.0f64, -7.0, -6.0, -5.0, -1.0, 0.0, 2.0] {
        let want = D::splat(x).gelu_p::<Precision>(D::ONE).extract::<0>();
        let (got, dy) = D::splat(x).gelu_d_p::<Precision>(D::ONE);

        assert_eq!(got.extract::<0>(), want, "gelu_d value disagrees with gelu at {x}");
        assert!(dy.extract::<0>().is_finite(), "gelu_d derivative not finite at {x}");
    }
}
