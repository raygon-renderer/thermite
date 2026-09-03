//! Correctness gate for the Riemann zeta pair, driven through the public
//! `RealSpecialMath::zeta` / `::zetac` entries, which are also the dispatched paths.
//!
//! The two functions are tested separately and against separately generated references,
//! because the whole point of `zetac` is that it is _not_ reachable from `zeta`: its table was
//! produced by summing `n^-s` from 2 at a working precision scaled to the answer's own
//! magnitude, since mpmath's `zeta(s) - 1` loses the value past `s = 100` exactly the way
//! binary64 does.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]
// Reference values are pasted from mpmath at full width.
#![allow(clippy::excessive_precision)]

use thermite::Vector;
use thermite::math::policy::policies::{MediumPrecision, Performance, Precision};
use thermite::prelude::*;
use thermite_special::SpecialMathWithPolicy;

type V = Vector<f64>;

fn zeta(s: f64) -> f64 {
    V::zeta_p::<Precision>(V::splat(s)).extract::<0>()
}

fn zetac(s: f64) -> f64 {
    V::zetac_p::<Precision>(V::splat(s)).extract::<0>()
}

fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    }
}

// (s, zeta(s)) from mpmath at 30 digits. Spans the critical strip, the region just past the
// pole, the flat tail, and negative s (which takes the functional equation).
const ZETA: &[(f64, f64)] = &[
    (0.1, -0.60303751985624172166),
    (0.5, -1.4603545088095868129),
    (0.9, -9.4301140194022545911),
    (1.5, 2.6123753486854883433),
    (2.0, 1.6449340668482264365),
    (2.5, 1.3414872572509171798),
    (3.0, 1.2020569031595942854),
    (4.0, 1.0823232337111381915),
    (7.0, 1.0083492773819228268),
    (12.0, 1.0002460865533080483),
    (25.0, 1.0000000298035035147),
    (40.0, 1.0000000000009094948),
    (-0.5, -0.20788622497735456602),
    (-1.0, -0.083333333333333333333),
    (-2.5, 0.0085169287778503305424),
    (-5.0, -0.003968253968253968254),
    (-11.0, 0.021092796092796092796),
];

// (s, zeta(s) - 1), summed directly from n = 2 at scaled precision. The last four rows are
// entirely unreachable by subtracting 1 from a binary64 zeta.
const ZETAC: &[(f64, f64)] = &[
    (2.0, 0.64493406684822643647),
    (5.0, 0.036927755143369926331),
    (10.0, 0.00099457512781808533715),
    (20.0, 9.5396203387279611315e-7),
    (40.0, 9.0949478402638892825e-13),
    (60.0, 8.6736173801199337283e-19),
    (80.0, 8.2718061255303444037e-25),
    (150.0, 7.0064923216240853546e-46),
    (400.0, 3.8725919148493182728e-121),
    (700.0, 1.9010915662951598235e-211),
];

#[test]
fn zeta_matches_reference() {
    for &(s, want) in ZETA {
        let got = zeta(s);
        // 1e-14 covers the critical strip, which is the weak region at ~13 ulp. Everywhere
        // else measures within a couple of ulp.
        assert!(rel(got, want) < 1e-14, "zeta({s}): got {got}, want {want}");
    }
}

#[test]
fn zetac_matches_reference() {
    for &(s, want) in ZETAC {
        let got = zetac(s);
        assert!(rel(got, want) < 1e-14, "zetac({s}): got {got}, want {want}");
    }
}

// The reason `zetac` exists. Past s ~ 40 the complement is below the mantissa of zeta itself,
// so the subtraction is not merely inaccurate but empty. This asserts both that our
// zetac keeps the value and that the subtraction really does lose it, so the test cannot
// quietly stop proving anything.
#[test]
fn zetac_is_not_reachable_by_subtracting_one() {
    for &(s, want) in &[
        (40.0f64, 9.0949478402638892825e-13f64),
        (80.0, 8.2718061255303444037e-25),
        (150.0, 7.0064923216240853546e-46),
        (700.0, 1.9010915662951598235e-211),
    ] {
        let direct = zetac(s);
        assert!(rel(direct, want) < 1e-14, "zetac({s}): got {direct}, want {want}");

        // 1e-9 rather than something tighter: at s = 40 the subtraction still has six good
        // digits (it is off by 9e-8), and only becomes total further out. The bar is
        // "meaningfully degraded", not "destroyed".
        let subtracted = zeta(s) - 1.0;
        assert!(
            rel(subtracted, want) > 1e-9,
            "at s={s} the subtraction was supposed to be ruined, but gave {subtracted} against \
             {want} - if this fails, zetac has stopped being necessary and the test should say so"
        );
    }
    // zeta itself is exactly 1 up there, which is the correct rounding of the true value.
    assert_eq!(zeta(200.0), 1.0, "zeta(200) rounds to 1");
    assert!(zetac(200.0) > 0.0, "zetac(200) must still be positive");
}

// zeta(2) = pi^2/6, zeta(4) = pi^4/90, zeta(-1) = -1/12, zeta(-3) = 1/120. Exact closed
// forms, independent of any table.
#[test]
fn matches_the_closed_forms() {
    use core::f64::consts::PI;

    let cases = [
        (2.0, PI * PI / 6.0),
        (4.0, PI * PI * PI * PI / 90.0),
        (6.0, PI.powi(6) / 945.0),
        (-1.0, -1.0 / 12.0),
        (-3.0, 1.0 / 120.0),
        (-5.0, -1.0 / 252.0),
    ];
    for (s, want) in cases {
        assert!(rel(zeta(s), want) < 1e-14, "zeta({s}) vs closed form {want}");
    }

    // The trivial zeros: zeta(-2n) = 0 for positive integer n. The functional equation gets
    // these from sin(pi s / 2), so they test that arm's sine rather than the series.
    for n in 1..=6 {
        let s = -2.0 * n as f64;
        let z = zeta(s);
        assert!(z.abs() < 1e-15, "zeta({s}) is a trivial zero, got {z}");
    }
}

#[test]
fn the_pole_and_its_neighbourhood() {
    assert_eq!(zeta(1.0), f64::INFINITY, "zeta(1) is a simple pole");

    // Approaching from the right, zeta(s) ~ 1/(s-1) + gamma. The boundary term carries the
    // divergence explicitly, so this should stay accurate arbitrarily close in.
    const EULER_GAMMA: f64 = 0.5772156649015328606;
    for &d in &[1e-4f64, 1e-6, 1e-8] {
        let got = zeta(1.0 + d);
        let want = 1.0 / d + EULER_GAMMA;
        assert!(
            rel(got, want) < 1e-4 * d.sqrt().max(1e-4),
            "zeta(1+{d}): got {got}, want ~{want}"
        );
    }
}

// Below `Average` the correction sum is truncated (8 terms -> 4 -> 2), so the tiers trade
// accuracy for a shorter ladder. Nothing else here runs off the default policy.
#[test]
fn lower_tiers_stay_within_their_budget() {
    for &(s, want) in ZETA {
        let at_default = V::zeta_p::<Performance>(V::splat(s)).extract::<0>();
        assert!(
            rel(at_default, want) < 1e-13,
            "zeta({s}) at Performance: got {at_default}"
        );

        let at_medium = V::zeta_p::<MediumPrecision<Precision>>(V::splat(s)).extract::<0>();
        assert!(!at_medium.is_nan(), "zeta({s}) went NaN at Medium");
        assert!(
            rel(at_medium, want) < 1e-9,
            "zeta({s}) at Medium: got {at_medium}, want {want}"
        );
    }
}
