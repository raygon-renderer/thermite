//! What the marker layer itself promises, beyond the kernels the family suites grade:
//! `Scaled<J>` / `Scaled<Y>` are the plain values on a real vector (unit scale factor), the
//! scalar surface exists for every entry and matches the vector spelling bit for bit, and
//! the runtime spherical order matches its const twin.

#![cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]

use thermite::Vector;
use thermite::math::policy::policies::Precision;
use thermite::prelude::*;
use thermite_special::bessel::{Ai, BesselOrder, BiPrime, I, J, K, Scaled, Y};
use thermite_special::{ScalarSpecialMath, SpecialMath, SpecialMathWithPolicy};

type D = Vector<f64>;

fn same_bits(what: &str, x: f64, got: f64, want: f64) {
    assert!(
        got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
        "{what}: x = {x}: {got:e} != {want:e}"
    );
}

fn grid() -> Vec<f64> {
    let mut xs = vec![
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        1.0,
        -1.0,
        2.0,
        0.5,
    ];
    let mut x = 1e-3;
    while x <= 300.0 {
        xs.push(x);
        xs.push(-x);
        x *= 1.9;
    }
    xs
}

#[test]
fn scaled_j_and_y_are_unit_scalings_on_the_real_axis() {
    type S = <D as GenericVector>::Signed;
    let orders = [
        BesselOrder::Integer(S::splat(3)),
        BesselOrder::HalfInteger(S::splat(5)),
        BesselOrder::Thirds(S::splat(1)),
        BesselOrder::Real(D::splat(7.4)),
    ];
    for &x in &grid() {
        let d = D::splat(x);
        same_bits(
            "jve_2",
            x,
            d.bessel_n_p::<Precision, Scaled<J>, 2>().extract::<0>(),
            d.bessel_n_p::<Precision, J, 2>().extract::<0>(),
        );
        same_bits(
            "yve_-1",
            x,
            d.bessel_n_p::<Precision, Scaled<Y>, -1>().extract::<0>(),
            d.bessel_n_p::<Precision, Y, -1>().extract::<0>(),
        );
        for order in orders {
            same_bits(
                "jve",
                x,
                d.bessel_p::<Precision, Scaled<J>>(order).extract::<0>(),
                d.bessel_p::<Precision, J>(order).extract::<0>(),
            );
            same_bits(
                "yve",
                x,
                d.bessel_p::<Precision, Scaled<Y>>(order).extract::<0>(),
                d.bessel_p::<Precision, Y>(order).extract::<0>(),
            );
        }
        same_bits(
            "sph jve_3",
            x,
            d.sph_bessel_n_p::<Precision, Scaled<J>, 3>().extract::<0>(),
            d.sph_bessel_n_p::<Precision, J, 3>().extract::<0>(),
        );
    }
}

#[test]
fn scalar_surface_matches_the_vector_spelling() {
    for &x in &grid() {
        let d = D::splat(x);
        same_bits(
            "J_2",
            x,
            x.scalar_bessel_n::<J, 2>(),
            d.bessel_n::<J, 2>().extract::<0>(),
        );
        same_bits(
            "e^x K_1",
            x,
            x.scalar_bessel_n::<Scaled<K>, 1>(),
            d.bessel_n::<Scaled<K>, 1>().extract::<0>(),
        );
        same_bits(
            "I_0.3",
            x,
            x.scalar_bessel::<I>(BesselOrder::Real(0.3)),
            d.bessel::<I>(BesselOrder::Real(D::splat(0.3))).extract::<0>(),
        );
        same_bits(
            "j_3",
            x,
            x.scalar_sph_bessel_n::<J, 3>(),
            d.sph_bessel_n::<J, 3>().extract::<0>(),
        );
        same_bits(
            "k_2",
            x,
            x.scalar_sph_bessel::<K>(2),
            d.sph_bessel::<K>(2).extract::<0>(),
        );
        same_bits("Ai", x, x.scalar_airy::<Ai>(), d.airy::<Ai>().extract::<0>());
        same_bits(
            "e^-z Bi'",
            x,
            x.scalar_airy::<Scaled<BiPrime>>(),
            d.airy::<Scaled<BiPrime>>().extract::<0>(),
        );
        let (a, ap, b, bp) = x.scalar_airy_all::<true>();
        let (va, vap, vb, vbp) = d.airy_all::<true>();
        same_bits("all Ai", x, a, va.extract::<0>());
        same_bits("all Ai'", x, ap, vap.extract::<0>());
        same_bits("all Bi", x, b, vb.extract::<0>());
        same_bits("all Bi'", x, bp, vbp.extract::<0>());
        let xf = x as f32;
        let got = xf.scalar_bessel_n::<Y, 1>();
        let want = Vector::<f32>::splat(xf).bessel_n::<Y, 1>().extract::<0>();
        assert!(
            got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
            "f32 Y_1 at {xf}"
        );
    }
}

#[test]
fn spherical_runtime_order_matches_const() {
    for &x in &grid() {
        let d = D::splat(x);
        same_bits(
            "j_4",
            x,
            d.sph_bessel_p::<Precision, J>(4).extract::<0>(),
            d.sph_bessel_n_p::<Precision, J, 4>().extract::<0>(),
        );
        same_bits(
            "e^x k_1",
            x,
            d.sph_bessel_p::<Precision, Scaled<K>>(1).extract::<0>(),
            d.sph_bessel_n_p::<Precision, Scaled<K>, 1>().extract::<0>(),
        );
        same_bits(
            "e^-x i_5",
            x,
            d.sph_bessel_p::<Precision, Scaled<I>>(5).extract::<0>(),
            d.sph_bessel_n_p::<Precision, Scaled<I>, 5>().extract::<0>(),
        );
    }
}

#[test]
fn oscillating_family_vanishes_at_infinity() {
    type S = <D as GenericVector>::Signed;
    let x = D::splat(f64::INFINITY);

    assert_eq!(x.bessel_n::<J, 0>().extract::<0>(), 0.0);
    assert_eq!(x.bessel_n::<J, 1>().extract::<0>(), 0.0);
    assert_eq!(x.bessel_n::<Y, 0>().extract::<0>(), 0.0);
    assert_eq!(x.bessel_n::<Y, 3>().extract::<0>(), 0.0);
    assert_eq!(x.sph_bessel_n::<J, 2>().extract::<0>(), 0.0);
    assert_eq!(x.sph_bessel::<Y>(3).extract::<0>(), 0.0);

    for order in [
        BesselOrder::HalfInteger(S::splat(5)),
        BesselOrder::Thirds(S::splat(1)),
        BesselOrder::Real(D::splat(2.25)),
        BesselOrder::Real(D::splat(-0.75)),
    ] {
        assert_eq!(x.bessel::<J>(order).extract::<0>(), 0.0, "J at inf, {order:?}");
        assert_eq!(x.bessel::<Y>(order).extract::<0>(), 0.0, "Y at inf, {order:?}");
    }
}
