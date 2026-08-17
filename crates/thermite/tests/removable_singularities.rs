//! `sqrt1mexp`, `versinc` and `logmean`: functions whose defining expression is
//! either `0/0` at a point or cancels to nothing near one.
//!
//! Each test does two things. It pins the value against a hand-computed constant, and
//! where the point is worth making, demonstrates that the direct spelling really does
//! fail on the same input. A rewrite that is merely equivalent is not worth a library
//! function, so these are here because the obvious form is wrong.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

use thermite::Vector;
use thermite::math::{RealMath, TranscendentalMath};
use thermite::prelude::*;

const E: f64 = core::f64::consts::E;
const LN_2: f64 = core::f64::consts::LN_2;
const PI: f64 = core::f64::consts::PI;

type D = Vector<f64>;
type F = Vector<f32>;

#[track_caller]
fn close(name: &str, got: f64, want: f64, tol: f64) {
    let rel = if want == 0.0 { got.abs() } else { ((got - want) / want).abs() };
    assert!(rel <= tol, "{name}: got {got:?}, want {want:?} (rel {rel:e})");
}

fn sqrt1mexp(x: f64) -> f64 {
    D::splat(x).sqrt1mexp().extract::<0>()
}
fn versinc(x: f64) -> f64 {
    D::splat(x).versinc().extract::<0>()
}
fn logmean(x: f64, y: f64) -> f64 {
    D::splat(x).logmean(D::splat(y)).extract::<0>()
}

// (phi_1 = (e^x - 1)/x, which used to open this file, moved to thermite-special
// with the rest of the phi-function family. See its tests/phi_functions.rs.)

// --- sqrt1mexp(x) = sqrt(1 - e^-x) ---

#[test]
fn sqrt1mexp_values_and_domain() {
    close("sqrt1mexp(0)", sqrt1mexp(0.0), 0.0, 0.0);
    close("sqrt1mexp(ln2)", sqrt1mexp(LN_2), 0.5_f64.sqrt(), 1e-15);
    close("sqrt1mexp(inf)", sqrt1mexp(f64::INFINITY), 1.0, 0.0);
    assert!(sqrt1mexp(-1.0).is_nan(), "negative argument is out of domain");
}

#[test]
fn sqrt1mexp_beats_the_direct_form() {
    // A thermostat friction of 1e-18 per step: the direct form gives exactly zero
    // noise, which silently turns a Langevin integrator into a deterministic one.
    let x = 1e-18_f64;
    let naive = (1.0 - (-x).exp()).sqrt();
    assert_eq!(naive, 0.0, "precondition: the direct form is expected to collapse here");

    close("sqrt1mexp tiny", sqrt1mexp(x), x.sqrt(), 1e-15);
}

// --- versinc(x) = (1 - cos x)/x^2 ---

#[test]
fn versinc_values_and_limits() {
    close("versinc(0)", versinc(0.0), 0.5, 0.0); // the removable singularity
    close("versinc(pi)", versinc(PI), 2.0 / (PI * PI), 1e-15);
    close("versinc(pi/2)", versinc(PI / 2.0), 1.0 / (PI * PI / 4.0), 1e-15);
    // 2pi: 1 - cos = 0 exactly, so the ratio is 0.
    close("versinc(2pi)", versinc(2.0 * PI), 0.0, 1e-15);
}

#[test]
fn versinc_beats_the_direct_form_near_zero() {
    // f32 rotation of 1e-4 radians, an entirely ordinary angular step. cos rounds to
    // exactly 1, so the direct Rodrigues coefficient is zero and the rotation matrix
    // silently loses its second-order term.
    let x = 1e-4_f32;
    let naive = (1.0 - x.cos()) / (x * x);
    assert_eq!(naive, 0.0, "precondition: the direct form is expected to collapse here");

    let got = F::splat(x).versinc().extract::<0>();
    close("versinc f32 tiny", got as f64, 0.5, 1e-6);
}

#[test]
fn versinc_matches_the_series_across_scales() {
    // 1/2 - x^2/24 + x^4/720 is good to well under a double ulp out to x = 1e-2.
    for k in 0..12 {
        let x = 1e-2 / (2.0_f64).powi(k);
        let series = 0.5 - x * x / 24.0 + x.powi(4) / 720.0;
        close("versinc series", versinc(x), series, 1e-14);
    }
}

// --- logmean(x, y) = (x - y)/(ln x - ln y) ---

#[test]
fn logmean_values_and_limit() {
    close("logmean(1, e)", logmean(1.0, E), E - 1.0, 1e-14);
    close("logmean(1, 2)", logmean(1.0, 2.0), 1.0 / LN_2, 1e-14);
    close("logmean(3, 3)", logmean(3.0, 3.0), 3.0, 0.0); // the 0/0 limit
    close("logmean symmetric", logmean(2.0, 7.0), logmean(7.0, 2.0), 1e-15);
}

#[test]
fn logmean_is_between_geometric_and_arithmetic() {
    // The defining inequality, and a decent smoke test that nothing is inverted.
    for &(x, y) in &[(1.0_f64, 2.0_f64), (0.5, 9.0), (1e-3, 1.0), (10.0, 10.5)] {
        let (g, a, l): (f64, f64, f64) = ((x * y).sqrt(), (x + y) / 2.0, logmean(x, y));
        assert!(g <= l && l <= a, "logmean({x}, {y}) = {l} not in [{g}, {a}]");
    }
}

#[test]
fn logmean_beats_the_direct_form_for_nearby_arguments() {
    // Two temperatures a part in 1e12 apart: ln x - ln y cancels to a couple of bits.
    let x = 300.0_f64;
    let y = x * (1.0 + 1e-12);
    let naive = (x - y) / (x.ln() - y.ln());

    let got = logmean(x, y);
    close("logmean nearby", got, 300.000_000_000_15, 1e-12);

    let naive_err = ((naive - got) / got).abs();
    assert!(
        naive_err > 1e-6,
        "precondition: the direct form is expected to be visibly wrong here, got {naive_err:e}"
    );
}

// --- the whole set on a wide backend, to be sure nothing is scalar-only ---

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn wide_backend_agrees_with_scalar() {
    use thermite::simd::Simd;
    type W = Vector<<thermite::backend::x86_v3::X86V3 as Simd>::f64x4>;

    let xs = [0.0, 1e-9, 0.5, 1.0, -2.0];
    let w = W::from_slice(&[xs[1], xs[2], xs[3], xs[4]]);

    let (s, v) = (w.sqrt1mexp(), w.versinc());
    let (s, v) = (s.into_array(), v.into_array());

    for i in 0..4 {
        let x = xs[i + 1];
        close("wide versinc", v.as_slice()[i], versinc(x), 1e-15);
        if x > 0.0 {
            close("wide sqrt1mexp", s.as_slice()[i], sqrt1mexp(x), 1e-15);
        }
    }

    let lm = W::splat(2.0).logmean(W::splat(7.0)).extract::<0>();
    close("wide logmean", lm, logmean(2.0, 7.0), 1e-15);
}
