//! Dumps a grid of special-function values as CSV for differential comparison against
//! `scipy.special`. Driven by `scripts/scipy_crosscheck.py`.
//!
//! Every reference table in this crate's test suite was generated with mpmath by the same
//! hand that wrote the kernels, which makes those tables *precise* but not *independent*: a
//! systematic misunderstanding of a definition, a normalization or a branch would agree with
//! itself at any number of digits. SciPy is a second, unrelated implementation, so this is
//! the check those tables cannot be.
//!
//!     cargo run --release -p thermite-special --example scipy_dump > scipy_grid.csv

use thermite::math::{RealMath, TranscendentalMath};
use thermite::prelude::*;
use thermite_complex::Complex;
use thermite_complex::prelude::ComplexSpecialMath;
use thermite_special::{RealSpecialMath, SpecialMath};

type D = Vector<f64>;
type C = Complex<D>;

fn row1(name: &str, a: f64, v: D) {
    println!("{name},{a:.17e},0,{:.17e},0", v.extract::<0>());
}

fn row2(name: &str, a: f64, b: f64, v: D) {
    println!("{name},{a:.17e},{b:.17e},{:.17e},0", v.extract::<0>());
}

fn rowc(name: &str, a: f64, b: f64, v: C) {
    println!(
        "{name},{a:.17e},{b:.17e},{:.17e},{:.17e}",
        v.re.extract::<0>(),
        v.im.extract::<0>()
    );
}

fn s(x: f64) -> D {
    D::splat(x)
}

/// A geometric-ish sweep that covers several decades without needing a huge grid.
fn decades(lo: f64, hi: f64, n: usize) -> impl Iterator<Item = f64> {
    let (l, h) = (lo.ln(), hi.ln());

    (0..n).map(move |i| (l + (h - l) * (i as f64) / ((n - 1) as f64)).exp())
}

fn main() {
    println!("fn,a,b,re,im");

    // --- error-function family -------------------------------------------------------
    for x in (-60..=60).map(|i| i as f64 * 0.15) {
        row1("erf", x, s(x).erf());
        row1("erfc", x, s(x).erfc());
        row1("erfcx", x, s(x).erfcx());
    }
    for x in decades(1e-8, 25.0, 40) {
        row1("erfcx", x, s(x).erfcx());
        row1("erfcx", -x, s(-x).erfcx());
    }
    for u in (1..=199).map(|i| i as f64 / 100.0 - 1.0) {
        row1("erfinv", u, s(u).erfinv());
    }
    for p in (1..=99).map(|i| i as f64 / 100.0) {
        row1("probit", p, s(p).probit());
    }

    // --- gamma family ----------------------------------------------------------------
    for x in (1..=400).map(|i| i as f64 * 0.05) {
        row1("tgamma", x, s(x).tgamma());
        row1("lgamma", x, s(x).lgamma());
        row1("digamma", x, s(x).digamma());
    }
    for x in (1..=60).map(|i| -(i as f64) * 0.37 - 0.011) {
        row1("tgamma", x, s(x).tgamma());
        row1("lgamma", x, s(x).lgamma());
        row1("digamma", x, s(x).digamma());
    }
    for x in decades(1e-6, 1e6, 30) {
        row1("lgamma", x, s(x).lgamma());
        row1("digamma", x, s(x).digamma());
    }
    for a in (1..=12).map(|i| i as f64 * 0.7) {
        for b in (1..=12).map(|j| j as f64 * 0.9) {
            row2("beta", a, b, s(a).beta(s(b)));
            row2("lbeta", a, b, s(a).lbeta(s(b)));
        }
    }

    // --- Lambert W, both branches ----------------------------------------------------
    for x in decades(1e-6, 1e8, 40) {
        let (w0, _) = s(x).lambert_w();
        row1("lambertw0", x, w0);
    }
    for x in (1..=40).map(|i| -(i as f64) * 0.009) {
        let (w0, wm1) = s(x).lambert_w();
        row1("lambertw0", x, w0);
        row1("lambertwm1", x, wm1);
    }

    // --- exponential integrals -------------------------------------------------------
    for x in decades(1e-3, 60.0, 40) {
        row1("expint1", x, s(x).expint::<1>());
        row1("expint2", x, s(x).expint::<2>());
        row1("expint3", x, s(x).expint::<3>());
    }

    // --- logistic / information theory ------------------------------------------------
    for x in (-80..=80).map(|i| i as f64 * 0.5) {
        row1("expit", x, s(x).logistic_sigmoid());
    }
    for p in (1..=999).map(|i| i as f64 / 1000.0) {
        row1("logit", p, s(p).logit());
    }
    for x in decades(1e-8, 1e3, 25) {
        row1("entr", x, s(x).entr());
        for y in decades(1e-6, 1e2, 8) {
            row2("xlogy", x, y, s(x).xlogy(s(y)));
            row2("xlog1py", x, y, s(x).xlog1py(s(y)));
            row2("rel_entr", x, y, s(x).rel_entr(s(y)));
            row2("kl_div", x, y, s(x).kl_div(s(y)));
        }
    }

    // --- power transforms ---------------------------------------------------------------
    for x in decades(1e-4, 1e4, 25) {
        for l in [-2.0, -0.5, 0.0, 1e-8, 0.25, 0.5, 1.0, 2.0, 3.5] {
            row2("boxcox", x, l, s(x).boxcox(s(l)));
            row2("boxcox1p", x, l, s(x).boxcox_1p(s(l)));
        }
    }
    for y in (-40..=40).map(|i| i as f64 * 0.7) {
        for l in [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0] {
            row2("yeojohnson", y, l, s(y).yeo_johnson(s(l)));
        }
    }
    for y in (-20..=40).map(|i| i as f64 * 0.31) {
        for l in [0.25, 0.5, 1.0, 2.0] {
            row2("inv_boxcox", y, l, s(y).inv_boxcox(s(l)));
            row2("inv_boxcox1p", y, l, s(y).inv_boxcox_1p(s(l)));
        }
    }

    // --- orthogonal polynomials --------------------------------------------------------
    for x in (-40..=40).map(|i| i as f64 * 0.1) {
        row1("hermite3", x, s(x).hermite::<3>());
        row1("hermite8", x, s(x).hermite::<8>());
        row1("legendre5", x, s(x).legendre(5, 0));
        row1("legendre7_2", x, s(x).legendre(7, 2));
    }
    for x in (0..=60).map(|i| i as f64 * 0.2) {
        row2("laguerre4", x, 0.0, s(x).laguerre::<4>(s(0.0)));
        row2("laguerre6a", x, 1.5, s(x).laguerre::<6>(s(1.5)));
    }
    for x in (-20..=20).map(|i| i as f64 * 0.05) {
        row2("jacobi4", x, 0.0, s(x).jacobi(s(0.5), s(1.5), 4, 0));
    }

    // --- Poisson ------------------------------------------------------------------------
    for k in [0.0, 1.0, 2.0, 5.0, 9.0, 20.0, 50.0, 200.0] {
        for lam in decades(0.1, 500.0, 12) {
            row2("poisson_pmf", k, lam, s(k).poisson_pmf(s(lam)));
        }
    }

    // --- complex: the Faddeeva core ------------------------------------------------------
    for xr in (-12..=12).map(|i| i as f64 * 0.7) {
        for yi in [0.0, 1e-6, 1e-3, 0.05, 0.3, 1.0, 3.0, 10.0] {
            let z = C::new(s(xr), s(yi));
            rowc("wofz", xr, yi, z.faddeeva_w());
            row1("voigt", 0.0, z.voigt());
            println!("voigt_xy,{xr:.17e},{yi:.17e},{:.17e},0", z.voigt().extract::<0>());
        }
    }
}
