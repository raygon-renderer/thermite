//! `sqrt1mexp`, `versinc`, `logmean`, `log1pmx`, `sinhc`, `cosh_m1` and `atanhc`: functions whose
//! defining expression is
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
    let rel = if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    };
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
fn log1pmx(x: f64) -> f64 {
    D::splat(x).log1pmx().extract::<0>()
}
fn log1pmx_f32(x: f32) -> f32 {
    F::splat(x).log1pmx().extract::<0>()
}
fn sinhc(x: f64) -> f64 {
    D::splat(x).sinhc().extract::<0>()
}
fn cosh_m1(x: f64) -> f64 {
    D::splat(x).cosh_m1().extract::<0>()
}
fn atanhc(x: f64) -> f64 {
    D::splat(x).atanhc().extract::<0>()
}

// (phi_1 = (e^x - 1)/x lives in thermite-special with the rest of the phi-function
// family. See its tests/phi_functions.rs.)

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

// --- log1pmx(x) = ln(1 + x) - x ---

/// mpmath at 50 digits, rounded once. Spread across the series window (`-1/2 <= x <= 1`),
/// both boundaries, and well outside it where the direct form runs.
#[rustfmt::skip]
const LOG1PMX: [(f64, f64); 14] = [
    (-0.75,             -0.6362943611198906),
    (-0.5,              -0.19314718055994531),
    (-0.4999999,        -0.19314708055996532),
    (-0.25,             -0.03768207245178093),
    (-0.0625,           -0.0020385211375711716),
    (-0.001,            -5.003335835335002e-07),
    ( 0.001,            -4.996669164668332e-07),
    ( 0.0625,           -0.0018753781835651575),
    ( 0.25,             -0.026856448685790246),
    ( 0.5,              -0.09453489189183562),
    ( 1.0,              -0.3068528194400547),
    ( 1.0000001,        -0.30685286944005596),
    ( 1.5,              -0.5837092681258449),
    ( 3.0,              -1.6137056388801094),
];

#[test]
fn log1pmx_matches_the_reference() {
    for &(x, want) in LOG1PMX.iter() {
        close(&format!("log1pmx({x})"), log1pmx(x), want, 4.0 * f64::EPSILON);
        close(
            &format!("f32 log1pmx({x})"),
            log1pmx_f32(x as f32) as f64,
            want,
            8.0 * f32::EPSILON as f64,
        );
    }
}

#[test]
fn log1pmx_beats_the_direct_form() {
    // A Poisson deviance at a rate 1e-17 from its mean: bd0 = -k * log1pmx(u). The direct
    // form does not merely lose digits here, it returns the wrong quantity: `1 + x` rounds
    // to exactly 1, so `ln(1 + x)` is 0 and the whole expression is `-x`, eighteen orders
    // of magnitude above the true -x^2/2, and with a deviance that should be negligible
    // reported as dominant.
    let x = 1e-17_f64;
    let naive = (1.0 + x).ln() - x;
    assert_eq!(
        naive, -x,
        "precondition: the direct form is expected to degenerate to -x here"
    );

    // -x^2/2 + x^3/3, and the cubic term is 67 orders below the square.
    close("log1pmx tiny", log1pmx(x), -0.5 * x * x, 1e-15);

    // At 1e-9 the direct form does not collapse, it just loses about half its digits -
    // 2 eps / |x| of relative error, which is the whole reason for the series.
    let x = 1e-9_f64;
    let want = -4.999999996666666e-19;
    close("log1pmx 1e-9", log1pmx(x), want, 4.0 * f64::EPSILON);
    let rel = (((1.0 + x).ln() - x) - want).abs() / want.abs();
    assert!(
        rel > 1e-8,
        "precondition: the direct form is expected to be inaccurate here, rel {rel:e}"
    );
}

#[test]
fn log1pmx_is_exact_at_zero_and_holds_its_domain() {
    assert_eq!(log1pmx(0.0), 0.0, "r = 0 makes the series identically zero");
    assert_eq!(log1pmx(-0.0), 0.0);
    assert_eq!(log1pmx(-1.0), f64::NEG_INFINITY, "ln 0 = -inf, +1 leaves -inf");
    assert!(log1pmx(-1.5).is_nan(), "x < -1 is out of domain");
    assert!(log1pmx(f64::NAN).is_nan());
    assert_eq!(log1pmx_f32(0.0), 0.0);
    assert_eq!(log1pmx_f32(-1.0), f32::NEG_INFINITY);
}

#[test]
fn log1pmx_is_continuous_across_the_series_window() {
    // The kernel switches from the odd series to `ln_1p(x) - x` at x = -1/2 and x = 1.
    // A window bug shows up as a step here and nowhere else, since each arm is smooth.
    for &(inside, outside, want_in, want_out) in &[
        (-0.5, -0.50000000000001, -0.19314718055994531, -0.1931471805599553),
        (1.0, 1.0000000000001, -0.3068528194400547, -0.3068528194401047),
    ] {
        close("window inside", log1pmx(inside), want_in, 4.0 * f64::EPSILON);
        close("window outside", log1pmx(outside), want_out, 4.0 * f64::EPSILON);
    }
}

#[test]
fn log1pmx_tracks_the_asymptote_across_scales() {
    // log1pmx(x) -> -x^2/2 as x -> 0, with a relative error of 2x/3 from the cubic term.
    let mut x = 1e-2_f64;
    while x > 1e-30 {
        for s in [1.0, -1.0] {
            let got = log1pmx(s * x);
            let want = -0.5 * x * x;
            close(&format!("asymptote at {}", s * x), got, want, 2.0 * x);
            assert!(got < 0.0, "log1pmx is negative everywhere but zero");
        }
        x *= 0.1;
    }
}

#[test]
fn log1pmx_holds_up_across_policies_and_on_the_scalar_surface() {
    use thermite::math::ScalarMath;
    use thermite::math::TranscendentalMathWithPolicy;
    use thermite::math::policy::policies::{Performance, Precision, UltraPerformance};

    // Average and above run the series. Medium and below return `ln_1p(x) - x` outright,
    // which on these arguments (none nearer zero than 1e-3) is itself good to a few ulp:
    // the direct form's loss is 2 eps / |x|, so it only becomes visible much closer in.
    for &(x, want) in LOG1PMX.iter() {
        let v = D::splat(x);
        close(
            &format!("Precision({x})"),
            v.log1pmx_p::<Precision>().extract::<0>(),
            want,
            4.0 * f64::EPSILON,
        );
        close(
            &format!("Performance({x})"),
            v.log1pmx_p::<Performance>().extract::<0>(),
            want,
            4.0 * f64::EPSILON,
        );
        // UltraPerformance is Worst + avoid_branching: the direct form, no window, no blend.
        let tol = (2.0 * f64::EPSILON / x.abs()).max(4.0 * f64::EPSILON);
        close(
            &format!("Ultra({x})"),
            v.log1pmx_p::<UltraPerformance>().extract::<0>(),
            want,
            tol,
        );
    }

    // Near zero the tiers separate, and the low one degrades on a schedule rather than off
    // a cliff: its relative error is the direct form's 2 eps / |x|, so it is still fine at
    // 1e-3, has lost four digits by 1e-12, and is meaningless once that quantity reaches 1.
    // Both halves are asserted: the bound it does keep, and the point where it has none.
    // Measured against `Precision`, not against -x^2/2: that asymptote is itself only good
    // to O(x), which would swamp the effect being measured at the top of the sweep.
    let mut x = 1e-3_f64;
    while x > 1e-13 {
        let want = D::splat(x).log1pmx_p::<Precision>().extract::<0>();
        close(
            "Performance tracks Precision",
            D::splat(x).log1pmx_p::<Performance>().extract::<0>(),
            want,
            4.0 * f64::EPSILON,
        );

        let ultra = D::splat(x).log1pmx_p::<UltraPerformance>().extract::<0>();
        let predicted = (8.0 * f64::EPSILON / x).max(8.0 * f64::EPSILON);
        close(&format!("Ultra at {x:e}"), ultra, want, predicted);
        x *= 1e-3;
    }

    // 2 eps / 1e-17 is about 44, so there is nothing left at all: the low tier returns a
    // value with no correct digit, while Average and up stay exact.
    let x = 1e-17_f64;
    let want = -0.5 * x * x;
    close(
        "Average at 1e-17",
        D::splat(x).log1pmx_p::<Performance>().extract::<0>(),
        want,
        1e-15,
    );
    close(
        "Precision at 1e-17",
        D::splat(x).log1pmx_p::<Precision>().extract::<0>(),
        want,
        1e-15,
    );
    // It lands on exactly 0 here, since its own ln_1p returns x unchanged and the
    // subtraction takes everything, which is 100% relative error, hence `>=` rather than `>`.
    let ultra = D::splat(x).log1pmx_p::<UltraPerformance>().extract::<0>();
    assert!(
        ((ultra - want) / want).abs() >= 1.0,
        "Worst tier is expected to have abandoned the near-zero form here, got {ultra:e}"
    );

    // The scalar_-prefixed surface on bare f64/f32 comes from the same decl_math! entry.
    close(
        "scalar f64",
        0.25_f64.scalar_log1pmx(),
        -0.026856448685790246,
        4.0 * f64::EPSILON,
    );
    close(
        "scalar f32",
        0.25_f32.scalar_log1pmx() as f64,
        -0.026856448685790246,
        8.0 * f32::EPSILON as f64,
    );
}

// --- sinhc(x) = sinh(x)/x and cosh_m1(x) = cosh(x) - 1 ---

/// mpmath at 40 digits, rounded once. `sinhc` is even, so the negative rows double as a
/// symmetry check against their positive twins.
#[rustfmt::skip]
const SINHC: [(f64, f64); 11] = [
    (-3.0,    3.3392916424699672),
    (-1.0,    1.1752011936438014),
    (-0.5,    1.0421906109874948),
    (1e-9,    1.0),
    (1e-4,    1.0000000016666666),
    (0.5,     1.0421906109874948),
    (1.0,     1.1752011936438014),
    (2.0,     1.8134302039235093),
    (5.0,     14.840642115557753),
    (10.0,    1101.3232874703394),
    (20.0,    12129129.885244757),
];

#[rustfmt::skip]
const COSH_M1: [(f64, f64); 10] = [
    (-1.0,    0.5430806348152438),
    (-0.5,    0.12762596520638078),
    (1e-9,    5e-19),
    (1e-8,    5e-17),
    (1e-4,    5.000000004166667e-09),
    (0.5,     0.12762596520638078),
    (1.0,     0.5430806348152438),
    (2.0,     2.7621956910836314),
    (5.0,     73.20994852478785),
    (20.0,    242582596.70489514),
];

#[test]
fn sinhc_matches_the_reference() {
    for &(x, want) in SINHC.iter() {
        close(&format!("sinhc({x})"), sinhc(x), want, 4.0 * f64::EPSILON);
        close(
            &format!("f32 sinhc({x})"),
            F::splat(x as f32).sinhc().extract::<0>() as f64,
            want,
            8.0 * f32::EPSILON as f64,
        );
    }
}

#[test]
fn cosh_m1_matches_the_reference() {
    for &(x, want) in COSH_M1.iter() {
        close(&format!("cosh_m1({x})"), cosh_m1(x), want, 4.0 * f64::EPSILON);
        close(
            &format!("f32 cosh_m1({x})"),
            F::splat(x as f32).cosh_m1().extract::<0>() as f64,
            want,
            8.0 * f32::EPSILON as f64,
        );
    }
}

#[test]
fn cosh_m1_beats_the_direct_form() {
    // cosh(x) - 1 is O(x^2) against a cosh of 1, so the subtraction eats the mantissa from
    // the top down: half the digits gone by 1e-4, all of them by 1e-8.
    let x = 1e-8_f64;
    let naive = x.cosh() - 1.0;
    assert_eq!(naive, 0.0, "precondition: the direct form is expected to collapse here");
    close("cosh_m1 tiny", cosh_m1(x), 5e-17, 1e-15);

    let x = 1e-4_f64;
    let want = 5.000000004166667e-09;
    close("cosh_m1 1e-4", cosh_m1(x), want, 4.0 * f64::EPSILON);
    let rel = ((x.cosh() - 1.0) - want).abs() / want;
    assert!(
        rel > 1e-10,
        "precondition: the direct form is expected to be inaccurate here, rel {rel:e}"
    );
}

#[test]
fn sinhc_and_cosh_m1_hold_their_limits() {
    // The removable singularity, and the limit sinhc has that sinc does not: sinh(x)/x
    // grows, and the naive spelling gets inf/inf = NaN at the ends rather than +inf.
    assert_eq!(sinhc(0.0), 1.0);
    assert_eq!(sinhc(-0.0), 1.0);
    assert_eq!(sinhc(f64::INFINITY), f64::INFINITY);
    assert_eq!(
        sinhc(f64::NEG_INFINITY),
        f64::INFINITY,
        "sinhc is even, so both ends are +inf"
    );
    assert!(sinhc(f64::NAN).is_nan());

    assert_eq!(cosh_m1(0.0), 0.0);
    assert_eq!(cosh_m1(-0.0), 0.0);
    assert_eq!(cosh_m1(f64::INFINITY), f64::INFINITY);
    assert_eq!(cosh_m1(f64::NEG_INFINITY), f64::INFINITY);

    assert_eq!(F::splat(0.0).sinhc().extract::<0>(), 1.0);
    assert_eq!(F::splat(0.0).cosh_m1().extract::<0>(), 0.0);
}

#[test]
fn sinhc_is_even_and_tracks_its_series() {
    // sinhc(x) = 1 + x^2/6 + x^4/120, with the quartic term below an ulp for x < 1e-2.
    let mut x = 1e-2_f64;
    while x > 1e-30 {
        let series = 1.0 + x * x / 6.0 + x.powi(4) / 120.0;
        close("sinhc series", sinhc(x), series, 1e-15);
        assert_eq!(sinhc(x), sinhc(-x), "sinhc is even");
        x *= 0.1;
    }

    // cosh_m1(x) = x^2/2 + x^4/24 + x^6/720. The sixth-order term is needed: at x = 1e-2 it
    // is 2.8e-11 of the total, so a two-term oracle fails a correct kernel here.
    let mut x = 1e-2_f64;
    while x > 1e-30 {
        let series = 0.5 * x * x + x.powi(4) / 24.0 + x.powi(6) / 720.0;
        close("cosh_m1 series", cosh_m1(x), series, 1e-15);
        assert_eq!(cosh_m1(x), cosh_m1(-x), "cosh_m1 is even");
        x *= 0.1;
    }
}

#[test]
fn einstein_heat_capacity_falls_out_of_sinhc() {
    // E(x) = x^2 e^x / (e^x - 1)^2 is exactly 1/sinhc(x/2)^2, which is the payoff: the
    // direct denominator overflows at x ~ 710 while E just underflows, and near zero it is
    // 0/0. Through sinhc it is stable across the whole range with no branch at all.
    for &x in &[1e-9_f64, 1e-4, 0.5, 1.0, 5.0, 20.0, 100.0, 700.0] {
        let s = sinhc(x * 0.5);
        let got = 1.0 / (s * s);

        // Reference by the direct formula, valid only where it does not overflow or cancel.
        if (1.0..=30.0).contains(&x) {
            let e = x.exp();
            let want = x * x * e / ((e - 1.0) * (e - 1.0));
            close(&format!("Einstein({x})"), got, want, 1e-12);
        }
        assert!(got > 0.0 && got <= 1.0 + 1e-15, "E({x}) = {got} out of (0, 1]");
    }
    // The limit at zero is exactly 1, where the direct form is 0/0.
    assert_eq!(1.0 / (sinhc(0.0) * sinhc(0.0)), 1.0);
}

// --- atanhc(x) = atanh(x)/x ---

/// mpmath at 40 digits. `atanhc` is even, so the negative rows double as a symmetry check.
#[rustfmt::skip]
const ATANHC: [(f64, f64); 11] = [
    (-0.99,  2.67338627511338),
    (-0.5,   1.0986122886681098),
    (-0.125, 1.0052577131236242),
    (1e-9,   1.0),
    (1e-4,   1.0000000033333334),
    (0.125,  1.0052577131236242),
    (0.5,    1.0986122886681098),
    (0.75,   1.2972734327035422),
    (0.9,    1.6357994328702448),
    (0.99,   2.67338627511338),
    (0.999,  3.8040051724226225),
];

#[test]
fn atanhc_matches_the_reference() {
    for &(x, want) in ATANHC.iter() {
        close(&format!("atanhc({x})"), atanhc(x), want, 4.0 * f64::EPSILON);
        close(
            &format!("f32 atanhc({x})"),
            F::splat(x as f32).atanhc().extract::<0>() as f64,
            want,
            16.0 * f32::EPSILON as f64,
        );
    }
}

#[test]
fn atanhc_holds_its_limits_and_domain() {
    assert_eq!(atanhc(0.0), 1.0, "the removable singularity");
    assert_eq!(atanhc(-0.0), 1.0);

    // atanh(+-1) is a signed infinity and the division by x restores the sign, so both ends
    // are +inf with no patch needed.
    assert_eq!(atanhc(1.0), f64::INFINITY);
    assert_eq!(atanhc(-1.0), f64::INFINITY);

    // Outside [-1, 1] atanh itself is NaN.
    assert!(atanhc(1.5).is_nan());
    assert!(atanhc(-2.0).is_nan());
    assert!(atanhc(f64::NAN).is_nan());

    assert_eq!(F::splat(0.0).atanhc().extract::<0>(), 1.0);
}

#[test]
fn atanhc_is_even_and_tracks_its_series() {
    // atanhc(x) = 1 + x^2/3 + x^4/5 + x^6/7, with the sixth-order term under an ulp by 1e-2.
    let mut x = 1e-2_f64;
    while x > 1e-30 {
        let series = 1.0 + x * x / 3.0 + x.powi(4) / 5.0 + x.powi(6) / 7.0;
        close("atanhc series", atanhc(x), series, 1e-15);
        assert_eq!(atanhc(x), atanhc(-x), "atanhc is even");
        x *= 0.1;
    }

    // atanhc >= 1 everywhere on the domain, with equality only at the origin.
    for &(x, _) in ATANHC.iter() {
        assert!(atanhc(x) >= 1.0, "atanhc({x}) = {} dropped below 1", atanhc(x));
    }
}

#[test]
fn logmean_is_the_reciprocal_of_atanhc() {
    // logmean(1+x, 1-x) = 2x/ln((1+x)/(1-x)) = x/atanh(x) = 1/atanhc(x). `logmean` is
    // written on `atanhc` for exactly this reason, so the identity pins the wiring.
    for &(x, want) in ATANHC.iter() {
        if x <= -1.0 || x >= 1.0 {
            continue;
        }
        // atanhc steepens as |x| -> 1 (its derivative carries a 1/(1-x^2)), so the identity
        // is correspondingly ill-conditioned there, so the tolerance scales with `1/(1-x^2)`
        // instead of being a flat number that only passes for the easy rows.
        let tol = 8.0 * f64::EPSILON / (1.0 - x * x);
        close(
            &format!("logmean(1+{x}, 1-{x})"),
            logmean(1.0 + x, 1.0 - x),
            1.0 / want,
            tol,
        );
    }

    // The general relation, on arguments away from the diagonal: with f = (a-b)/(a+b),
    // logmean(a, b) = (a+b)/(2 atanhc(f)).
    for &(a, b) in &[(1.0_f64, 2.0_f64), (0.5, 9.0), (300.0, 301.0), (1e-3, 1.0)] {
        let f = (a - b) / (a + b);
        close("logmean via atanhc", logmean(a, b), (a + b) / (2.0 * atanhc(f)), 1e-14);
    }

    // And the equal-argument limit still lands on the value itself, which is now the
    // arithmetic mean via atanhc(0) = 1 rather than a separately patched 0/0.
    close("logmean(3,3)", logmean(3.0, 3.0), 3.0, 0.0);
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

    // Mixed lanes on purpose: two inside the log1pmx series window, two outside it, so
    // the blend between the two arms is exercised rather than a uniform fast path.
    let mixed = [-0.75, -0.25, 0.5, 3.0];
    let lp = W::from_slice(&mixed).log1pmx().into_array();
    for (i, &x) in mixed.iter().enumerate() {
        close("wide log1pmx", lp.as_slice()[i], log1pmx(x), 4.0 * f64::EPSILON);
    }

    // Likewise for sinhc: two lanes inside the Taylor window, two outside.
    let mixed = [1e-6, -1e-5, 0.5, 3.0];
    let sh = W::from_slice(&mixed).sinhc().into_array();
    let cm = W::from_slice(&mixed).cosh_m1().into_array();
    for (i, &x) in mixed.iter().enumerate() {
        close("wide sinhc", sh.as_slice()[i], sinhc(x), 4.0 * f64::EPSILON);
        close("wide cosh_m1", cm.as_slice()[i], cosh_m1(x), 4.0 * f64::EPSILON);
    }

    // atanhc, staying inside its [-1, 1] domain, two lanes each side of the window.
    let mixed = [1e-6, -1e-5, 0.5, -0.9];
    let at = W::from_slice(&mixed).atanhc().into_array();
    for (i, &x) in mixed.iter().enumerate() {
        close("wide atanhc", at.as_slice()[i], atanhc(x), 4.0 * f64::EPSILON);
    }
}
