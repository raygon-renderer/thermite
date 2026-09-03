//! `bessel_ratio::<I>` (`A_nu = I_nu / I_{nu-1}`, the vMF mean resultant length) and its
//! inverse, against mpmath (`scripts/bessel_ratio_ref.py`, 50 digits. The inverse rows are
//! the exact inverses of their f64 arguments by `findroot`).
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]

use thermite::math::policy::DefaultPolicy;
use thermite::math::policy::policies::BestPrecision;
use thermite::prelude::*;
use thermite_special::bessel::{I, Scaled};
use thermite_special::{RealSpecialMath, RealSpecialMathWithPolicy};

include!("common/wide.rs");

include!("bessel_ratio_ref/table.rs");

type D = Vector<f64>;
type F = Vector<f32>;
type Best = BestPrecision<DefaultPolicy>;

fn ulps(got: f64, want: f64, eps: f64) -> f64 {
    if got == want {
        return 0.0;
    }
    if !got.is_finite() || !want.is_finite() {
        return f64::INFINITY;
    }
    ((got - want) / want).abs() / eps
}

#[test]
fn ratio_f64() {
    for (name, f) in [
        (
            "ratio best",
            (|nu: f64, x: f64| D::splat(x).bessel_ratio_p::<Best, I>(D::splat(nu)).extract::<0>())
                as fn(f64, f64) -> f64,
        ),
        ("ratio", |nu, x| {
            D::splat(x).bessel_ratio::<I>(D::splat(nu)).extract::<0>()
        }),
    ] {
        let mut w = (0.0f64, 0.0, 0.0);
        for &(nu, x, want) in RATIO.iter() {
            let got = f(nu, x);
            // Past x = 1e4 the quotient arm carries `bessel_iv`'s own accuracy in its
            // asymptotic region (16 ulp at 1e5, 1e-14 at 1e6 per the Bessel notes).
            let u = ulps(got, want, f64::EPSILON) / if x >= 1e4 { 4.0 } else { 1.0 };
            assert!(u.is_finite(), "{name}(nu={nu}, x={x}): got {got:e}, want {want:e}");
            if u > w.0 {
                w = (u, nu, x);
            }
        }
        eprintln!("{name}: worst {:.2} ulp at nu = {}, x = {}", w.0, w.1, w.2);
        assert!(w.0 <= 8.0, "{name}: {w:?}");
    }
}

/// Scored against the inverse's own condition number, `2 kappa eps / (p - 1)` relative per
/// ulp of `r` (the docs' caveat): the table's `kappa` is the exact inverse of the f64 `r`,
/// so this is the kernel's error alone, and must stay a few ulp _times_ that factor,
/// the forward's ulp error, which is the Bessel kernel's 16 past `x = 1e4`.
#[test]
fn inverse_f64() {
    for (name, f) in [
        (
            "inverse best",
            (|nu: f64, r: f64| D::splat(r).inv_bessel_ratio_p::<Best, I>(D::splat(nu)).extract::<0>())
                as fn(f64, f64) -> f64,
        ),
        ("inverse", |nu, r| {
            D::splat(r).inv_bessel_ratio::<I>(D::splat(nu)).extract::<0>()
        }),
    ] {
        let mut w = (0.0f64, 0.0, 0.0);
        for &(nu, r, kappa) in INV_RATIO.iter() {
            let got = f(nu, r);
            let cond = (1.0 + 2.0 * kappa / (2.0 * nu - 1.0)).max(1.0) * if kappa >= 1e4 { 4.0 } else { 1.0 };
            let u = ulps(got, kappa, f64::EPSILON) / cond;
            assert!(u.is_finite(), "{name}(nu={nu}, r={r}): got {got:e}, want {kappa:e}");
            if u > w.0 {
                w = (u, nu, r);
            }
        }
        eprintln!("{name}: worst {:.2} scaled ulp at nu = {}, r = {}", w.0, w.1, w.2);
        assert!(w.0 <= 8.0, "{name}: {w:?}");
    }
}

#[test]
fn f32_default() {
    let eps = f32::EPSILON as f64;
    let mut w = 0.0f64;
    for &(nu, x, want) in RATIO.iter() {
        let got = F::splat(x as f32).bessel_ratio::<I>(F::splat(nu as f32)).extract::<0>() as f64;
        w = w.max(ulps(got, want, eps));
    }
    eprintln!("f32 ratio worst {w:.2} ulp");
    assert!(w <= 16.0, "{w}");
}

/// Round trips through the forward, and the two elementary cases: `nu = 1` against
/// `I_1/I_0` from the integer kernels, `nu = 3/2` against the Langevin function.
#[test]
fn identities() {
    use thermite_special::SpecialMath;

    for &(nu, r, _) in INV_RATIO.iter() {
        let kappa = D::splat(r).inv_bessel_ratio::<I>(D::splat(nu));
        let back = kappa.bessel_ratio::<I>(D::splat(nu)).extract::<0>();
        assert!(
            ulps(back, r, f64::EPSILON) <= 16.0,
            "A(inv_A({r})) at nu = {nu}: {back}"
        );
    }
    for &x in &[0.1f64, 1.0, 3.0, 10.0, 50.0] {
        let v = D::splat(x);
        let via_int = (v.bessel_n::<Scaled<I>, 1>() / v.bessel_n::<Scaled<I>, 0>()).extract::<0>();
        assert!(
            ulps(v.bessel_ratio::<I>(D::splat(1.0)).extract::<0>(), via_int, f64::EPSILON) <= 16.0,
            "nu = 1 at {x}"
        );
        let lang = v.langevin().extract::<0>();
        assert!(
            ulps(v.bessel_ratio::<I>(D::splat(1.5)).extract::<0>(), lang, f64::EPSILON) <= 16.0,
            "nu = 3/2 at {x}"
        );
    }
}

#[test]
fn edges() {
    let d = |v: f64| D::splat(v);
    let nu = d(2.5);
    assert_eq!(d(0.0).bessel_ratio::<I>(nu).extract::<0>(), 0.0);
    assert_eq!(d(-0.0).bessel_ratio::<I>(nu).extract::<0>(), 0.0);
    assert_eq!(d(f64::INFINITY).bessel_ratio::<I>(nu).extract::<0>(), 1.0);
    assert_eq!(
        d(-3.0).bessel_ratio::<I>(nu).extract::<0>(),
        -d(3.0).bessel_ratio::<I>(nu).extract::<0>()
    );
    assert!(d(f64::NAN).bessel_ratio::<I>(nu).extract::<0>().is_nan());

    assert_eq!(d(0.0).inv_bessel_ratio::<I>(nu).extract::<0>(), 0.0);
    assert_eq!(d(1.0).inv_bessel_ratio::<I>(nu).extract::<0>(), f64::INFINITY);
    assert!(d(1.5).inv_bessel_ratio::<I>(nu).extract::<0>().is_nan());
    assert_eq!(
        d(-0.3).inv_bessel_ratio::<I>(nu).extract::<0>(),
        -d(0.3).inv_bessel_ratio::<I>(nu).extract::<0>()
    );
    // Below 1e-8 the answer is p r outright, to r^3.
    assert!(ulps(d(1e-9).inv_bessel_ratio::<I>(nu).extract::<0>(), 5e-9, f64::EPSILON) <= 1.0);
}

/// The complement `1 - A` in the tail, where the plain form has already rounded to zero
/// or nearly so, and its inverse scored flat: the complement form has no `1/t` condition.
#[test]
fn complement_f64() {
    let mut w = (0.0f64, 0.0, 0.0);
    for &(nu, x, want) in RATIO_1M.iter() {
        let got = D::splat(x).bessel_ratio_1m::<I>(D::splat(nu)).extract::<0>();
        // Below x = 8 nu it is 1 - A: within 8 eps of a complement of at least 1/8, and
        // within 2x eps/(2 nu - 1) in the corner 8 nu <= x < 20. Above, direct.
        let corner = if x < 20.0 {
            (2.0 * x / (2.0 * nu - 1.0)).max(8.0)
        } else {
            4.0
        };
        let u = ulps(got, want, f64::EPSILON) / corner;
        assert!(u.is_finite(), "complement(nu={nu}, x={x}): got {got:e}, want {want:e}");
        if u > w.0 {
            w = (u, nu, x);
        }
    }
    eprintln!("complement: worst {:.2} scaled ulp at nu = {}, x = {}", w.0, w.1, w.2);
    assert!(w.0 <= 8.0, "{w:?}");

    let mut w = (0.0f64, 0.0, 0.0);
    for &(nu, t, kappa) in INV_RATIO_1M.iter() {
        let got = D::splat(t).inv_bessel_ratio_1m::<I>(D::splat(nu)).extract::<0>();
        let u = ulps(got, kappa, f64::EPSILON);
        assert!(
            u.is_finite(),
            "inv complement(nu={nu}, t={t}): got {got:e}, want {kappa:e}"
        );
        if u > w.0 {
            w = (u, nu, t);
        }
    }
    eprintln!("inverse complement: worst {:.2} ulp at nu = {}, t = {}", w.0, w.1, w.2);
    assert!(w.0 <= 16.0, "{w:?}");

    // Odd symmetry through the complement, and the two ends.
    let nu = D::splat(2.5);
    assert_eq!(
        D::splat(-30.0).bessel_ratio_1m::<I>(nu).extract::<0>(),
        2.0 - D::splat(30.0).bessel_ratio_1m::<I>(nu).extract::<0>()
    );
    assert_eq!(D::splat(0.0).bessel_ratio_1m::<I>(nu).extract::<0>(), 1.0);
    assert_eq!(D::splat(f64::INFINITY).bessel_ratio_1m::<I>(nu).extract::<0>(), 0.0);
    assert_eq!(D::splat(0.0).inv_bessel_ratio_1m::<I>(nu).extract::<0>(), f64::INFINITY);
    assert_eq!(D::splat(1.0).inv_bessel_ratio_1m::<I>(nu).extract::<0>(), 0.0);
    assert!(D::splat(-0.1).inv_bessel_ratio_1m::<I>(nu).extract::<0>().is_nan());
}

/// Every arm in one packet, each lane bit-identical to a splat of itself.
#[test]
fn packets_mix_arms_bit_exactly() {
    let nus = f64x4::new([1.0, 1.5, 25.0, 150.0]);
    let xs = [0.5f64, 3.0, 40.0, 1e4];
    let got = f64x4::new(xs).bessel_ratio::<I>(nus).into_array();
    let nu_a = nus.into_array();
    for k in 0..4 {
        let s = f64x4::splat(xs[k])
            .bessel_ratio::<I>(f64x4::splat(nu_a[k]))
            .extract::<0>();
        assert_eq!(got[k].to_bits(), s.to_bits(), "ratio lane {k}");
    }
    let rs = [0.05f64, 0.5, 0.9, 0.999];
    let got = f64x4::new(rs).inv_bessel_ratio::<I>(nus).into_array();
    for k in 0..4 {
        let s = f64x4::splat(rs[k])
            .inv_bessel_ratio::<I>(f64x4::splat(nu_a[k]))
            .extract::<0>();
        assert_eq!(got[k].to_bits(), s.to_bits(), "inverse lane {k}");
    }
}
