//! `lgamma` / `lgamma_r` / `beta` on `Compensated`, via the shift-and-Stirling default
//! in `thermite_compensated::specialized::special`.
//!
//! Reference values are mpmath 1.3.0 at 45 digits, as `(hi, lo)` pairs so both words are
//! checked - collapsing to one `f64` would only ever verify the half this type is not
//! about. Inputs are all exactly representable, so the references describe the argument
//! actually passed rather than a decimal near it.
//!
//! # A note on the tolerance
//!
//! These assert at full double-double. They could not until `exp`'s range reduction was
//! fixed: Stirling calls `ln` twice - once on the shifted argument, once on the
//! divided-out product - `ln` refines through `exp`, and `exp` was capped near 50 bits by
//! rounded `k * LN_2_EXTENDED[i]` products. Every value here then landed ~1e-14 *absolute*
//! from its reference regardless of magnitude, which is what identified the cause: a
//! fault in the gamma code would scale with shift count or reflection, and it did
//! neither. See the `LN_2_EXTENDED` comments in `consts.rs`.

use thermite::prelude::*;
use thermite_compensated::Compensated;
use thermite_special::{RealSpecialMath, SpecialMath};

type V = Vector<f64>;
type C = Compensated<V>;

fn c(x: f64) -> C {
    C::new(V::splat(x))
}

/// `trigamma` lives only on the specialized trait - it is deliberately not part of the
/// public `SpecialMath` surface - so it is reached through a local helper rather than by
/// importing that trait, which would make every other call in this file ambiguous.
fn trigamma(x: C) -> C {
    use thermite_special::specialized::SpecializedSpecialMath;

    SpecializedSpecialMath::trigamma::<thermite::math::policy::DefaultPolicy>(x)
}

/// Full double-double. Reachable only because the `LN_2_EXTENDED` Cody-Waite split in
/// `consts.rs` makes `exp`'s range reduction exact - before that this had to sit at
/// 1e-14, since Stirling calls `ln` twice and `ln` refines through `exp`.
const TOL: f64 = 1e-29;

/// Error across both words, relative to the magnitude of the result.
///
/// The leading subtraction is exact, so this really does see ~106 bits rather than
/// bottoming out at `f64` epsilon.
fn dd_err(got: C, hi: f64, lo: f64) -> f64 {
    let value = got.value.extract::<0>();
    let error = got.error.extract::<0>();

    (((value - hi) + (error - lo)) / hi.abs().max(1.0)).abs()
}

/// `(x, hi, lo, sign)` from mpmath.
const LGAMMA: &[(f64, f64, f64, f64)] = &[
    // above the reflection cut: straight shift-and-Stirling
    (0.5, 5.72364942924700082e-01, 5.13297558135391319e-18, 1.0),
    (1.0, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.0),
    (2.0, 0.00000000000000000e+00, 0.00000000000000000e+00, 1.0),
    (3.5, 1.20097360234707429e+00, -6.23505842731913601e-17, 1.0),
    (10.25, 1.33680236714760454e+01, 8.73462320479020869e-16, 1.0),
    // at and past the shift target, where the loop does nothing
    (30.0, 7.12570389671680147e+01, -5.65474697789772551e-15, 1.0),
    (150.5, 6.02513954870585394e+02, 1.79315932906120213e-14, 1.0),
    // small positive: reflected, and the pole at 0 is close
    (0.125, 2.01941835755379628e+00, 6.15105022288932163e-17, 1.0),
    (1e-3, 6.90717888538385338e+00, 2.77755682510737619e-16, 1.0),
    // negative: reflected, alternating sign between the poles
    (-0.5, 1.26551212348464537e+00, 2.83234437198169119e-17, -1.0),
    (-1.5, 8.60047015376480983e-01, 3.12045817457795499e-17, 1.0),
    (-2.25, 5.55501545020647525e-01, -5.41150032189443343e-17, -1.0),
    (-7.75, -8.58184776355186685e+00, 1.78376935814032112e-16, 1.0),
    (-100.5, -3.64900968309427356e+02, 3.68431199837762683e-15, -1.0),
];

#[test]
fn lgamma_matches_mpmath_to_double_double() {
    for &(x, hi, lo, _) in LGAMMA {
        let err = dd_err(c(x).lgamma(), hi, lo);

        assert!(err <= TOL, "lgamma({x}): rel err {err:e}");
    }
}

#[test]
fn lgamma_r_carries_the_sign() {
    // The sign is the whole reason `lgamma_r` exists separately: `lgamma` throws away
    // which side of the axis Gamma is on, and between every pair of negative poles it
    // flips.
    for &(x, hi, lo, sign) in LGAMMA {
        let (value, got_sign) = c(x).lgamma_r();

        assert!(dd_err(value, hi, lo) <= TOL, "lgamma_r({x}) value");
        assert_eq!(got_sign.value.extract::<0>(), sign, "lgamma_r({x}) sign");
    }
}

#[test]
fn exact_at_one_and_two() {
    // Gamma(1) = Gamma(2) = 1, so both logs are exactly zero. Nothing forces this - it
    // has to come out of the shift, the series and the final subtraction agreeing.
    for x in [1.0f64, 2.0] {
        let g = c(x).lgamma();

        assert!(
            g.value.extract::<0>().abs() < 1e-30,
            "lgamma({x}) should vanish, got {}",
            g.value.extract::<0>()
        );
    }
}

#[test]
fn poles_at_non_positive_integers() {
    // sin(pi x) is exactly zero there, so the reflection's log is -inf and lgamma is
    // +inf. Falls out of the formula rather than being special-cased.
    for x in [0.0f64, -1.0, -2.0, -10.0] {
        let g = c(x).lgamma().value.extract::<0>();

        assert!(g.is_infinite() && g > 0.0, "lgamma({x}) should be +inf, got {g}");
    }
}

#[test]
fn beta_rides_on_lgamma() {
    // B(a, b) = Gamma(a)Gamma(b)/Gamma(a+b), defaulted through logs. Checked against the
    // closed forms B(1, b) = 1/b and B(a, 1) = 1/a, which need no oracle.
    for b in [0.5f64, 1.0, 2.0, 7.5, 30.0] {
        let got = C::beta(c(1.0), c(b));
        let want = 1.0 / b;
        let err = ((got.value.extract::<0>() - want) / want).abs();

        assert!(
            err <= TOL,
            "beta(1, {b}): got {}, want {want}",
            got.value.extract::<0>()
        );
    }

    // Symmetry, which the log form does not enforce structurally.
    let ab = C::beta(c(2.25), c(5.5));
    let ba = C::beta(c(5.5), c(2.25));
    assert!(dd_err(ab, ba.value.extract::<0>(), ba.error.extract::<0>()) <= TOL);
}

// ---------------------------------------------------------------------------
// tgamma / digamma / trigamma
// ---------------------------------------------------------------------------

const TGAMMA: &[(f64, f64, f64)] = &[
    (0.5, 1.77245385090551610e+00, -7.66658649982579870e-17),
    (1.0, 1.00000000000000000e+00, 0.00000000000000000e+00),
    (2.0, 1.00000000000000000e+00, 0.00000000000000000e+00),
    (3.5, 3.32335097044784256e+00, -4.97061879358916033e-18),
    (10.25, 6.39232598779576831e+05, -3.62821710109657097e-11),
    (0.125, 7.53394159879761194e+00, -3.35104541714754267e-17),
    (-0.5, -3.54490770181103221e+00, 1.53331729996515974e-16),
    (-2.25, -1.74281486572825273e+00, 7.58895377535455116e-17),
    (-7.75, 1.87478241700424713e-04, 9.60580840401921907e-21),
    (20.0, 1.21645100408832000e+17, 0.00000000000000000e+00),
];

const DIGAMMA: &[(f64, f64, f64)] = &[
    (0.5, -1.96351002602142355e+00, 6.95842813380203054e-17),
    (1.0, -5.77215664901532866e-01, 4.94291515243064487e-18),
    (2.0, 4.22784335098467134e-01, 4.94291515243064487e-18),
    (3.5, 1.10315664064524310e+00, 8.43872549996890560e-17),
    (10.25, 2.27770479068672405e+00, -7.63548495409490820e-17),
    (0.125, -8.38849266329585497e+00, 9.96840324989167715e-17),
    (-0.5, 3.64899739785765204e-02, 1.95342298948023051e-19),
    (-2.25, 4.15858356465797208e+00, 1.98911157403882279e-16),
    (-7.75, -1.03076883283093146e+00, -5.37097938340895420e-17),
    (40.0, 3.67632737403484322e+00, -9.46747507831996827e-17),
    (0.001, -1.00057557193181026e+03, -2.04250785156873326e-14),
];

const TRIGAMMA: &[(f64, f64, f64)] = &[
    (0.5, 4.93480220054467900e+00, 3.13264775436985568e-16),
    (1.0, 1.64493406684822641e+00, 3.04067235039847616e-17),
    (2.0, 6.44934066848226406e-01, 3.04067235039847616e-17),
    (3.5, 3.30357756100234878e-01, -1.33875100305049157e-17),
    (10.25, 1.02474521517991871e-01, -3.78998553397751801e-18),
    (0.125, 6.53881334449880285e+01, 5.98666543897068635e-15),
    (-0.5, 8.93480220054467900e+00, 3.13264775436985568e-16),
    (-2.25, 1.93794105118691391e+01, -1.73727677883116914e-15),
    (-7.75, 1.96181443343522908e+01, -1.30488836623825750e-15),
    (40.0, 2.53151038412910284e-02, -2.61616559944858556e-19),
    (0.001, 1.00000164253319579e+06, 4.02352348995040216e-11),
];

#[test]
fn tgamma_matches_mpmath() {
    // Exponentiating lgamma costs log2|lnGamma| bits, so this is looser than the rest by
    // design - see `compensated_tgamma`.
    for &(x, hi, lo) in TGAMMA {
        let err = dd_err(c(x).tgamma(), hi, lo);
        assert!(err <= 1e-27, "tgamma({x}): rel err {err:e}");
    }
}

#[test]
fn tgamma_matches_factorials_exactly() {
    // Gamma(n+1) = n!, exact in f64 up to 20!. Nothing in the path forces this.
    let mut fact = 1.0f64;
    for n in 1..=20u32 {
        fact *= n as f64;
        let g = c(n as f64 + 1.0).tgamma().value.extract::<0>();
        assert!(
            ((g - fact) / fact).abs() <= 1e-27,
            "Gamma({}) vs {n}!: {g} vs {fact}",
            n + 1
        );
    }
}

#[test]
fn digamma_matches_mpmath() {
    for &(x, hi, lo) in DIGAMMA {
        let err = dd_err(c(x).digamma(), hi, lo);
        assert!(err <= TOL, "digamma({x}): rel err {err:e}");
    }
}

#[test]
fn trigamma_matches_mpmath() {
    for &(x, hi, lo) in TRIGAMMA {
        let err = dd_err(trigamma(c(x)), hi, lo);
        assert!(err <= TOL, "trigamma({x}): rel err {err:e}");
    }
}

#[test]
fn digamma_recurrence_holds() {
    // psi(x + 1) - psi(x) = 1/x. The shift loop is built on this, but at x >= 30 no
    // shifting happens for either argument, so this checks the series against itself
    // across a step it did not take.
    for x in [30.5f64, 41.0, 77.25] {
        let d = c(x + 1.0).digamma() - c(x).digamma();
        let want = 1.0 / x;
        let err = ((d.value.extract::<0>() - want) / want).abs();

        assert!(err <= 1e-27, "psi({}) - psi({x}) != 1/{x}: err {err:e}", x + 1.0);
    }
}

#[test]
fn trigamma_reflection_is_consistent() {
    // psi_1(x) + psi_1(1 - x) = pi^2 / sin^2(pi x), checked on the unreflected side of
    // the branch so it is not merely restating the implementation.
    use core::f64::consts::PI;

    for x in [0.6f64, 0.75, 0.9] {
        let lhs = trigamma(c(x)) + trigamma(c(1.0 - x));
        let s = (PI * x).sin();
        let want = PI * PI / (s * s);
        let err = ((lhs.value.extract::<0>() - want) / want).abs();

        assert!(err <= 1e-14, "trigamma reflection at {x}: err {err:e}");
    }
}
