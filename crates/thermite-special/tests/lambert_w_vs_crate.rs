//! `lambert_w` accuracy at the `Reference` tier, graded against mpmath at 40 digits,
//! with the `lambert_w` crate measured alongside as a second opinion.
//!
//! This also records *why* that crate is not the reference implementation for a function
//! libm has no entry point for. It documents 50 bits of accuracy and f64 carries 53, so
//! it is ~8 ulp by construction, and worse than that in the small-argument regime.
//! Thermite refines a piecewise seed with Halley against `exp` - which at this tier is
//! libm's - and Halley converges to the true root regardless of the seed, so the tier's
//! accuracy is inherited from the primitive rather than from any table.
//!
//! Keep the assertion: it fails if the crate ever becomes the better choice.
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use thermite::Vector;
use thermite::math::policy::policies::Reference;
use thermite::prelude::*;
use thermite_special::SpecialMathWithPolicy;

/// (x, W_0(x) to 25 significant digits, from mpmath with dps=40)
const EXACT: &[(f64, &str)] = &[
    (0.5, "0.3517337112491958260249093"),
    (1.0, "0.5671432904097838729999687"),
    (2.0, "0.8526055020137254913464724"),
    (10.0, "1.745528002740699383074301"),
    (100.0, "3.385630140290050184888244"),
    (1e6, "11.38335808614005262200016"),
    (-0.1, "-0.1118325591589629648335695"),
    (-0.3, "-0.4894022271802149690362313"),
    (-0.36, "-0.8060843159708177782855214"),
    (0.001, "0.0009990014973385308899578279"),
];

/// Error in ulp of the correctly rounded f64.
fn ulp_err(got: f64, exact_str: &str) -> f64 {
    let exact: f64 = exact_str.parse().unwrap();
    if got == exact {
        return 0.0;
    }
    let ulp = {
        let b = exact.abs().to_bits();
        f64::from_bits(b + 1) - exact.abs()
    };
    ((got - exact) / ulp).abs()
}

#[test]
fn thermite_reference_vs_lambert_w_crate() {
    let mut worst_thermite: f64 = 0.0;
    let mut worst_crate: f64 = 0.0;

    println!("{:>10}  {:>14}  {:>14}", "x", "thermite ulp", "crate ulp");
    for &(x, exact) in EXACT {
        let t = Vector::<f64>::splat(x).lambert_w_p::<Reference>().0.extract::<0>();
        let c = lambert_w::lambert_w0(x);

        let (te, ce) = (ulp_err(t, exact), ulp_err(c, exact));
        worst_thermite = worst_thermite.max(te);
        worst_crate = worst_crate.max(ce);

        println!("{x:>10}  {te:>14.2}  {ce:>14.2}");
    }

    println!("\nworst: thermite {worst_thermite:.2} ulp, crate {worst_crate:.2} ulp");

    // The claim under test: adopting the crate would not improve the tier.
    assert!(
        worst_thermite <= worst_crate.max(2.0),
        "thermite {worst_thermite} ulp is worse than the crate's {worst_crate} ulp - \
         the crate would be an upgrade after all"
    );
}
