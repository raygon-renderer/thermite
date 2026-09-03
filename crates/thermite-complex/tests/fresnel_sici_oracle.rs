//! `fresnel` and `sici` differenced against two functions this workspace already
//! ships, through completely different code.
//!
//! Both integrals are, on a ray in the complex plane, a function that exists here
//! already (verified against mpmath at 43 and 45 digits respectively):
//!
//! ```text
//! C(x) + i S(x)     = (1+i)/2 * erf(sqrt(pi)(1-i)x/2)
//! E_1(ix)           = -Ci(x) + i (Si(x) - pi/2)
//! ```
//!
//! Neither identity needs the auxiliary functions, and (the point) neither needs the
//! phase `pi x^2/2` reconstructed in the test, so this is a genuinely independent
//! lowering rather than the same arithmetic spelled twice. The real kernels sum
//! Chebyshev and Horner series over real coefficient tables. Complex `erf` runs the
//! Weideman/Faddeeva rational, and complex `E_1` an interleaved series and continued
//! fraction. Nothing is shared but the answer.
//!
//! This lives here rather than in `thermite-special` because the dependency runs this
//! way: `thermite-complex` imports the Weideman tables _from_ `thermite-special`, so
//! the special crate cannot reach the complex one at any price.
//!
//! The ranges straddle both crossovers (2.5265 for Fresnel, 12 for sici), which is
//! where a coefficient or branch-selection mistake would show first.
//!
//! # This test found a bug in complex `erf`, and then verified its fix
//!
//! `z = sqrt(pi)(1-i)x/2` walks a 45-degree ray, and the regime guard inside
//! `math/special/erf.rs` used to select its Taylor series there on a criterion
//! (`-Re(z^2)`) that is identically zero on that ray however large `z` grows, so the
//! series ran at a cancellation of `e^{2 Im(z)^2}`. Measured against this oracle before
//! the fix: 2.3e-11 at x = 3, 3.9e-9 at 3.5, 2.1e-6 at 4, and `erf` returning -3.0e10
//! at |z| = 8 where `|erf| <= 1.2`.
//!
//! The guard now tests `2 Im(z)^2`. This file covers the range that exposed it.
//! `erf_wedge.rs` is the direct regression test for the fix. This one is what found it.

#![cfg(feature = "special")]

use thermite::prelude::*;
use thermite_complex::Complex;
use thermite_complex::prelude::{ComplexSpecialMath, ComplexSpecialMathWithPolicy};
use thermite_special::{RealSpecialMath, SpecialMath, SpecialMathWithPolicy};

use thermite::math::policy::policies::Precision;

type V = Vector<f64>;
type C = Complex<V>;

fn c(re: f64, im: f64) -> C {
    Complex::new(V::splat(re), V::splat(im))
}

fn parts(z: C) -> (f64, f64) {
    (z.re.extract::<0>(), z.im.extract::<0>())
}

/// Still set by the oracle rather than the kernel: `fresnel` holds 2.8 ulp against
/// mpmath (6e-16), while complex `erf` runs the Weideman rational and is documented at
/// ~1e-13 normwise at this tier. The figures above are what this bound looked like
/// before the `erf` fix: 2.3e-11 at x = 3 and 6.9e-1 at x = 5.
const FRESNEL_ORACLE_TOL: f64 = 1e-12;

#[track_caller]
fn agree(what: &str, x: f64, got: f64, want: f64, scale: f64, tol: f64) {
    let err = (got - want).abs() / scale;
    assert!(
        err <= tol,
        "{what} at x = {x}: real kernel {got:e}, complex oracle {want:e} (rel {err:e} > {tol:e})"
    );
}

/// `C + iS = (1+i)/2 erf(sqrt(pi)(1-i)x/2)`.
///
/// `z` sits in the fourth quadrant, so `w(iz)` is in the upper half plane, exactly
/// where the Weideman approximation behind complex `erf` is valid. That is not a
/// coincidence: it is why the Fresnel auxiliaries could have been built out of the
/// Faddeeva machinery instead of their own tables, and what makes this a free
/// oracle rather than a second implementation someone had to write.
#[test]
fn fresnel_matches_complex_erf_on_the_45_degree_ray() {
    let k = core::f64::consts::PI.sqrt() / 2.0;
    for &x in &[
        0.05, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 2.5265, 2.53, 2.8, 3.0, 4.0, 5.0, 6.0, 8.0,
    ] {
        let (gc, gs) = V::splat(x).fresnel();
        let (gc, gs) = (gc.extract::<0>(), gs.extract::<0>());

        // (1+i)/2 * erf(z), z = sqrt(pi)(1-i)x/2
        let (er, ei) = parts(c(k * x, -k * x).erf_p::<Precision>());
        let (wc, ws) = ((er - ei) / 2.0, (er + ei) / 2.0);

        // C and S stay in [0.32, 0.72] past the first oscillation and are ~x below it,
        // so scale by the value itself. Both are strictly positive for x > 0.
        agree("C", x, gc, wc, wc.abs(), FRESNEL_ORACLE_TOL);
        agree("S", x, gs, ws, ws.abs(), FRESNEL_ORACLE_TOL);
    }
}

/// `E_1(ix) = -Ci(x) + i (Si(x) - pi/2)`.
///
/// The imaginary part is graded against `pi/2` rather than against `Si - pi/2`: that
/// difference decays like `1/x` and is a cancellation of the oracle's own making, not
/// something either kernel controls.
#[test]
fn sici_matches_complex_expint_on_the_imaginary_axis() {
    for &x in &[0.25, 0.75, 1.5, 3.0, 6.0, 9.0, 11.5, 12.0, 12.5, 15.0, 20.0] {
        let (gsi, gci) = V::splat(x).sici();
        let (gsi, gci) = (gsi.extract::<0>(), gci.extract::<0>());

        let (er, ei) = parts(c(0.0, x).expint_n::<1>());

        agree("Ci", x, gci, -er, er.abs().max(1.0 / x), 1e-12);
        agree(
            "Si",
            x,
            gsi,
            ei + core::f64::consts::FRAC_PI_2,
            core::f64::consts::FRAC_PI_2,
            1e-12,
        );
    }
}

/// The same two identities at the one place a differential test is most likely to
/// catch something: immediately either side of each crossover, where the two branches
/// of the real kernel meet and must agree with each other as well as with the oracle.
#[test]
fn the_branches_meet_continuously() {
    let k = core::f64::consts::PI.sqrt() / 2.0;
    for &(x0, eps) in &[(2.5265f64, 1e-9), (12.0, 1e-9)] {
        for &d in &[-eps, eps] {
            let x = x0 + d;
            if x0 < 5.0 {
                let (gc, gs) = V::splat(x).fresnel();
                let (er, ei) = parts(c(k * x, -k * x).erf_p::<Precision>());
                agree(
                    "C across",
                    x,
                    gc.extract::<0>(),
                    (er - ei) / 2.0,
                    1.0,
                    FRESNEL_ORACLE_TOL,
                );
                agree(
                    "S across",
                    x,
                    gs.extract::<0>(),
                    (er + ei) / 2.0,
                    1.0,
                    FRESNEL_ORACLE_TOL,
                );
            } else {
                let (gsi, gci) = V::splat(x).sici();
                let (er, ei) = parts(c(0.0, x).expint_n::<1>());
                agree("Ci across", x, gci.extract::<0>(), -er, 1.0 / x, 1e-12);
                agree(
                    "Si across",
                    x,
                    gsi.extract::<0>(),
                    ei + core::f64::consts::FRAC_PI_2,
                    core::f64::consts::FRAC_PI_2,
                    1e-12,
                );
            }
        }
    }
}
