//! Correctness gate for the Pochhammer symbol, driven through the public
//! `RealSpecialMath::pochhammer` entry, which is also the dispatched path.
//!
//! The three internal paths have very different error characters, so the tests separate
//! them: the exact-product path (small integer `m`) is held to a much tighter tolerance than
//! the Stirling-difference path, whose error necessarily tracks `|ln (z)_m| * eps` because it
//! exponentiates a logarithm. The identity tests are independent of the reference table and
//! are what would catch a path being selected wrongly.
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
use thermite_special::RealSpecialMathWithPolicy;

type V = Vector<f64>;

fn poch(z: f64, m: f64) -> f64 {
    V::pochhammer_p::<Precision>(V::splat(z), V::splat(m)).extract::<0>()
}

fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    }
}

// (z, m, (z)_m) from mpmath at 30 digits.
const POCH: &[(f64, f64, f64)] = &[
    (1.0, 5.0, 120.0),
    (3.0, 1.0, 3.0),
    (0.5, 3.0, 1.875),
    (2.0, 0.0, 1.0),
    (0.001, 1.0, 0.0010000000000000000208),
    (3.7, 100.0, 5.9056116821724653401e+162),
    (50.0, 17.0, 8.9488931244344544524e+29),
    (200.0, 2.0, 40200.0),
    (10000.0, 1.0, 10000.0),
    (100000000.0, 0.0001, 1.0018437657235250801),
    (0.5, 0.5, 0.56418958354775628695),
    (2.5, -2.0, 1.3333333333333333333),
    (-3.5, 2.0, 8.75),
    (5.0, -2.0, 0.083333333333333333333),
    (-0.5, 0.25, 1.3827346780725867011),
    (9.0, 0.5, 2.9586424105805805329),
    (100000000.0, 0.5, 9999.9999875000000078),
    (0.1, 30.0, 1.3039350096613176474e+30),
];

#[test]
fn matches_reference() {
    for &(z, m, want) in POCH {
        let got = poch(z, m);
        // 1e-12 covers the Stirling path's worst case (z = 3.7, m = 100, result near 1e170,
        // where the exp-of-a-logarithm floor is ~1.2e-13). The integer path is checked far
        // more tightly below.
        assert!(rel(got, want) < 1e-12, "({z})_{m}: got {got}, want {want}");
    }
}

// The case that motivates the whole kernel. `exp(lgamma(z+m) - lgamma(z))` has no correct
// digits here. Anything near the true value proves the naive form is not being used.
#[test]
fn small_m_beside_large_z_does_not_cancel() {
    let got = poch(1e8, 1e-4);
    let want = 1.0018437657235250801;
    assert!(rel(got, want) < 1e-14, "(1e8)_1e-4: got {got}, want {want}");

    let naive = (libm::lgamma(1e8 + 1e-4) - libm::lgamma(1e8)).exp();
    assert!(
        rel(naive, want) > 1e-8,
        "the naive form was supposed to be bad here, got {naive} - if this fails the test is \
         no longer proving anything"
    );
}

// Small integer m takes the exact product, which forms no logarithm and should be within a
// few ulp. Checked against the product built independently in the test.
#[test]
fn integer_m_is_the_exact_product() {
    for &z in &[0.25f64, 1.0, 2.5, 7.0, -3.5, -0.5, 100.0, 1e5] {
        for n in 0..=20u32 {
            let mut want = 1.0f64;
            for j in 0..n {
                want *= z + j as f64;
            }
            let got = poch(z, n as f64);
            assert!(
                rel(got, want) < 1e-14,
                "({z})_{n}: got {got}, want {want} (relative {})",
                rel(got, want)
            );
        }
    }
}

// The poles: a product spanning a non-positive integer contains an exact zero.
#[test]
fn zero_factors_are_exact() {
    assert_eq!(poch(-2.0, 3.0), 0.0, "(-2)_3 spans z = 0");
    assert_eq!(poch(-1.0, 5.0), 0.0, "(-1)_5 spans z = 0");
    assert_eq!(poch(0.0, 2.0), 0.0, "(0)_2 starts at zero");
    assert_eq!(poch(-3.0, 2.0), 6.0, "(-3)_2 = (-3)(-2), no zero spanned");
}

// (z)_{m+1} = (z)_m (z + m) holds identically, and does not depend on the reference table.
// Crossing the product cap at m = 20 also checks that the two paths agree where they meet.
#[test]
fn satisfies_the_recurrence() {
    for &z in &[0.5f64, 1.0, 3.7, 40.0] {
        for m in 0..25u32 {
            let a = poch(z, m as f64);
            let b = poch(z, m as f64 + 1.0);
            assert!(
                rel(b, a * (z + m as f64)) < 1e-12,
                "({z})_{} != ({z})_{m} * ({z} + {m})",
                m + 1
            );
        }
    }
}

// (z)_m (z+m)_k = (z)_{m+k} for real, non-integer steps. Exercises the Stirling path on
// both sides and is independent of any table.
#[test]
fn splits_across_real_steps() {
    for &(z, m, k) in &[(1.5f64, 0.25f64, 0.75f64), (3.0, 2.5, 1.5), (12.0, 0.5, 30.5)] {
        let split = poch(z, m) * poch(z + m, k);
        let whole = poch(z, m + k);
        assert!(
            rel(split, whole) < 1e-12,
            "({z})_{m} ({}) _{k} != ({z})_{}",
            z + m,
            m + k
        );
    }
}

// m = 0 is exactly 1 everywhere it is defined, including where z is negative.
#[test]
fn empty_product_is_one() {
    for &z in &[0.5f64, -2.5, 1e8, -1e3, 1.0] {
        assert_eq!(poch(z, 0.0), 1.0, "({z})_0");
    }
}

// Extreme arguments must saturate, never produce NaN. This used to need a guard: shifting
// both arguments by a shared amount walked `z + m` far past where it needed to go, `(z+m)^9`
// overflowed, and `exp(huge) * 0` gave NaN. Shifting each argument only as far as it needs
// makes every product bounded by 18^9, so `exp` saturating is already the right answer and
// there is nothing to repair. Kept as a regression test on that structure.
#[test]
fn extreme_arguments_saturate_instead_of_producing_nan() {
    for &(z, m) in &[
        (0.001f64, 1e300f64),
        (0.001, 1e35),
        (0.5, 1e40),
        (0.001, 300.0),
        (0.5, 700.0),
        (1e-8, 1e5),
    ] {
        let got = poch(z, m);
        assert!(!got.is_nan(), "({z})_{m} produced NaN");
        assert_eq!(got, f64::INFINITY, "({z})_{m}: got {got}, want inf");
    }
}

// The mirror direction: a result far below the smallest normal underflows to zero rather
// than to NaN.
#[test]
fn far_underflow_reaches_zero() {
    for &(z, m) in &[(1e5f64, -1e5 + 0.5f64), (300.0, -299.5)] {
        let got = poch(z, m);
        assert!(!got.is_nan(), "({z})_{m} produced NaN");
        assert!(
            (0.0..1e-30).contains(&got),
            "({z})_{m}: got {got}, expected a tiny positive"
        );
    }
}

// The exact-product path is gated at `Average`, so the boundary runs between the default
// policy (which takes it) and the tier below (which does not). Every other test here runs at
// `Precision`, so this is the one that pins both sides of that line.
#[test]
fn the_product_path_boundary_sits_below_the_default_policy() {
    fn at(z: f64, m: f64) -> f64 {
        V::pochhammer_p::<Performance>(V::splat(z), V::splat(m)).extract::<0>()
    }
    fn below(z: f64, m: f64) -> f64 {
        V::pochhammer_p::<MediumPrecision<Precision>>(V::splat(z), V::splat(m)).extract::<0>()
    }

    for &(z, m, want) in POCH {
        for (name, got) in [("Performance", at(z, m)), ("Medium", below(z, m))] {
            assert!(!got.is_nan(), "({z})_{m} went NaN at {name}");
            assert!(rel(got, want) < 1e-9, "({z})_{m} at {name}: got {got}, want {want}");
        }
    }

    // At the default the product path is live, so integer answers are exact.
    for &(z, n, want) in &[(3.0f64, 1.0f64, 3.0f64), (1.0, 5.0, 120.0), (200.0, 2.0, 40200.0)] {
        assert_eq!(at(z, n), want, "({z})_{n} should be exact at the default policy");
    }

    // Below the gate they are merely correct, which is the documented cost of the tier.
    assert!(rel(below(200.0, 2.0), 40200.0) < 1e-13, "(200)_2 below the gate");

    // Either side, the poles have to come out as zeros: below the gate that goes through the
    // logarithmic residue instead of a zero factor in the product.
    for (name, f) in [("Performance", at as fn(f64, f64) -> f64), ("Medium", below)] {
        assert_eq!(f(-2.0, 3.0), 0.0, "(-2)_3 at {name}");
        assert_eq!(f(0.0, 2.0), 0.0, "(0)_2 at {name}");
        assert!(rel(f(-3.5, 2.0), 8.75) < 1e-9, "(-3.5)_2 at {name}");
    }
}
