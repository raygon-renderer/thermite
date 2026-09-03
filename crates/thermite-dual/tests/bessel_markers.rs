//! The marker entries on `Dual`: const and runtime spherical orders agree in value and
//! derivative, and the scaled oscillating markers are the plain values on a real line.

#![cfg(feature = "special")]

use thermite::prelude::*;
use thermite_dual::Dual;
use thermite_special::SpecialMath;
use thermite_special::bessel::{I, J, K, Scaled, Y};

type V = Vector<f64>;
type D = Dual<V, 1>;

fn d(x: f64) -> D {
    D::variable(V::splat(x), 0)
}

fn same(what: &str, x: f64, got: D, want: D) {
    let (g, w) = (got.re.extract::<0>(), want.re.extract::<0>());
    assert!(g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()), "{what} value at {x}: {g:e} != {w:e}");
    let (g, w) = (got.dual[0].extract::<0>(), want.dual[0].extract::<0>());
    assert!(g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()), "{what} derivative at {x}: {g:e} != {w:e}");
}

const XS: [f64; 9] = [0.0, 1e-3, 0.5, 1.0, 2.5, 7.0, 15.0, 40.0, -3.0];

#[test]
fn dual_spherical_runtime_matches_const() {
    for &x in &XS {
        same("j_3", x, d(x).sph_bessel::<J>(3), d(x).sph_bessel_n::<J, 3>());
        same("y_2", x, d(x).sph_bessel::<Y>(2), d(x).sph_bessel_n::<Y, 2>());
        same("e^-x i_4", x, d(x).sph_bessel::<Scaled<I>>(4), d(x).sph_bessel_n::<Scaled<I>, 4>());
        same("e^x k_1", x, d(x).sph_bessel::<Scaled<K>>(1), d(x).sph_bessel_n::<Scaled<K>, 1>());
    }
}

#[test]
fn dual_scaled_j_is_j_on_the_real_line() {
    for &x in &XS {
        same("jve_2", x, d(x).bessel_n::<Scaled<J>, 2>(), d(x).bessel_n::<J, 2>());
        same("yve_1", x, d(x).bessel_n::<Scaled<Y>, 1>(), d(x).bessel_n::<Y, 1>());
    }
}
