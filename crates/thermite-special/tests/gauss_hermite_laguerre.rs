//! Gauss-Hermite and Gauss-Laguerre nodes and weights against mpmath roots
//! (`scripts/bessel_ratio_ref.py`, bracketed bisection at 50 digits), one root per lane.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]

use thermite::prelude::*;
use thermite_special::RealSpecialMath;

include!("bessel_ratio_ref/table.rs");

type D = Vector<f64>;
type F = Vector<f32>;

fn rel(got: f64, want: f64, eps: f64) -> f64 {
    ((got - want) / want).abs() / eps
}

/// Nodes relatively (they are not confined to a unit interval), weights relatively against
/// their condition number in the node. At a root of `H_n`, `H''/H' = 2x` and
/// `d ln w / dx = -4x` (weight `1/h_{n-1}^2`, `h_{n-1}'/h_{n-1} = 2x` at the root). At a root of
/// `L_n^alpha`, `L''/L' = (x - alpha - 1)/x` and `d ln w / dx = -1/x - 2 L''/L'`.
#[test]
fn hermite_f64() {
    let mut wx = (0.0f64, 0, 0);
    let mut ww = (0.0f64, 0, 0);
    for &(n, k, x, w) in GAUSS_HERMITE.iter() {
        let (gx, gw) = D::splat(k as f64).gauss_hermite(n);
        let ex = if x == 0.0 {
            gx.extract::<0>().abs() / f64::EPSILON
        } else {
            rel(gx.extract::<0>(), x, f64::EPSILON) * x.abs().min(1.0)
        };
        let cond = 1.0 + 4.0 * x.abs() * x.abs().max(1.0);
        let ew = rel(gw.extract::<0>(), w, f64::EPSILON) / cond;
        assert!(ex <= 8.0, "node n={n} k={k}: {} vs {x}, {ex:.1} ulp", gx.extract::<0>());
        assert!(
            ew <= 8.0,
            "weight n={n} k={k}: {} vs {w}, {ew:.1} scaled ulp",
            gw.extract::<0>()
        );
        if ex > wx.0 {
            wx = (ex, n, k);
        }
        if ew > ww.0 {
            ww = (ew, n, k);
        }
    }
    eprintln!(
        "hermite nodes worst {:.2} ulp at n={} k={}; weights worst {:.2} scaled ulp at n={} k={}",
        wx.0, wx.1, wx.2, ww.0, ww.1, ww.2
    );
}

#[test]
fn laguerre_f64() {
    let mut wx = (0.0f64, 0, 0);
    let mut ww = (0.0f64, 0, 0);
    for &(n, alpha, k, x, w) in GAUSS_LAGUERRE.iter() {
        let (gx, gw) = D::splat(k as f64).gauss_laguerre(D::splat(alpha), n);
        // Absolute below 1: the smallest roots sit at the recurrence's noise floor, an ulp of 1.
        let ex = rel(gx.extract::<0>(), x, f64::EPSILON) * x.min(1.0);
        let ddl = (x - alpha - 1.0) / x;
        let cond = 1.0 + (1.0 / x + 2.0 * ddl.abs()) * x.max(1.0);
        let ew = rel(gw.extract::<0>(), w, f64::EPSILON) / cond;
        // The noise floor of the recurrence grows with the size of the intermediate L_m, ~n^2.
        assert!(
            ex <= 8.0 * (1.0 + n as f64 / 32.0),
            "node n={n} a={alpha} k={k}: {} vs {x}, {ex:.1} ulp",
            gx.extract::<0>()
        );
        assert!(
            ew <= 8.0,
            "weight n={n} a={alpha} k={k}: {} vs {w}, {ew:.1} scaled ulp",
            gw.extract::<0>()
        );
        if ex > wx.0 {
            wx = (ex, n, k);
        }
        if ew > ww.0 {
            ww = (ew, n, k);
        }
    }
    eprintln!(
        "laguerre nodes worst {:.2} ulp at n={} k={}; weights worst {:.2} scaled ulp at n={} k={}",
        wx.0, wx.1, wx.2, ww.0, ww.1, ww.2
    );
}

/// Exactness on monomials: sum w x^m = Gamma((m+1)/2) for even m (Hermite, 0 for odd) and
/// Gamma(m + alpha + 1) (Laguerre), for m < 2n.
#[test]
fn integrate_monomials_exactly() {
    for n in [1u32, 2, 3, 5, 8, 16, 24] {
        let mut hs = vec![0.0f64; 2 * n as usize];
        let mut ls = vec![0.0f64; 2 * n as usize];
        for k in 0..n {
            let (x, w) = D::splat(k as f64).gauss_hermite(n);
            let (x, w) = (x.extract::<0>(), w.extract::<0>());
            let mut xm = 1.0;
            for s in hs.iter_mut() {
                *s += w * xm;
                xm *= x;
            }
            let (x, w) = D::splat(k as f64).gauss_laguerre(D::splat(0.5), n);
            let (x, w) = (x.extract::<0>(), w.extract::<0>());
            let mut xm = 1.0;
            for s in ls.iter_mut() {
                *s += w * xm;
                xm *= x;
            }
        }
        for m in 0..2 * n as usize {
            let want_h = if m % 2 == 0 {
                libm::tgamma((m as f64 + 1.0) / 2.0)
            } else {
                0.0
            };
            let scale_h = libm::tgamma((m as f64 + 1.0) / 2.0);
            assert!(
                (hs[m] - want_h).abs() <= 64.0 * f64::EPSILON * n as f64 * scale_h,
                "hermite n={n} m={m}: {} vs {want_h}",
                hs[m]
            );
            let want_l = libm::tgamma(m as f64 + 1.5);
            assert!(
                rel(ls[m], want_l, f64::EPSILON) <= 64.0 * n as f64,
                "laguerre n={n} m={m}: {} vs {want_l}",
                ls[m]
            );
        }
    }
}

#[test]
fn edges_and_f32() {
    let (x, w) = D::splat(0.0).gauss_hermite(1);
    assert_eq!(x.extract::<0>(), 0.0);
    assert!(rel(w.extract::<0>(), core::f64::consts::PI.sqrt(), f64::EPSILON) <= 1.0);
    let (x, w) = D::splat(0.0).gauss_laguerre(D::splat(2.0), 1);
    assert_eq!((x.extract::<0>(), w.extract::<0>()), (3.0, 2.0));
    assert!(D::splat(4.0).gauss_hermite(4).0.extract::<0>().is_nan());
    assert!(
        D::splat(0.0)
            .gauss_laguerre(D::splat(-1.5), 4)
            .0
            .extract::<0>()
            .is_nan()
    );
    // Mirror symmetry of the Hermite rule.
    for n in [4u32, 9, 20] {
        for k in 0..n {
            let (x, w) = D::splat(k as f64).gauss_hermite(n);
            let (x2, w2) = D::splat((n - 1 - k) as f64).gauss_hermite(n);
            assert_eq!(x.extract::<0>(), -x2.extract::<0>(), "n={n} k={k}");
            assert_eq!(w.extract::<0>(), w2.extract::<0>(), "n={n} k={k}");
        }
    }

    let mut worst = 0.0f64;
    for &(n, k, x, w) in GAUSS_HERMITE.iter() {
        if n > 32 {
            continue;
        }
        let (gx, gw) = F::splat(k as f32).gauss_hermite(n);
        let ex = if x == 0.0 {
            gx.extract::<0>().abs() as f64 / f32::EPSILON as f64
        } else {
            rel(gx.extract::<0>() as f64, x, f32::EPSILON as f64) * x.abs().min(1.0)
        };
        let cond = 1.0 + 4.0 * x.abs() * x.abs().max(1.0);
        let ew = rel(gw.extract::<0>() as f64, w, f32::EPSILON as f64) / cond;
        worst = worst.max(ex).max(ew);
        assert!(
            ex <= 8.0 && ew <= 8.0,
            "f32 hermite n={n} k={k}: node {ex:.1}, weight {ew:.1}"
        );
    }
    for &(n, alpha, k, x, w) in GAUSS_LAGUERRE.iter() {
        // L_{n-1}^2 at the largest root passes f32's range near n = 20.
        if n > 16 {
            continue;
        }
        let (gx, gw) = F::splat(k as f32).gauss_laguerre(F::splat(alpha as f32), n);
        let ex = rel(gx.extract::<0>() as f64, x, f32::EPSILON as f64) * x.min(1.0);
        let ddl = (x - alpha - 1.0) / x;
        let cond = 1.0 + (1.0 / x + 2.0 * ddl.abs()) * x.max(1.0);
        let ew = rel(gw.extract::<0>() as f64, w, f32::EPSILON as f64) / cond;
        worst = worst.max(ex).max(ew);
        assert!(
            ex <= 8.0 && ew <= 8.0,
            "f32 laguerre n={n} a={alpha} k={k}: node {ex:.1}, weight {ew:.1}"
        );
    }
    eprintln!("f32 worst {worst:.2}");
}

/// A packet of consecutive indices is the rule: every lane bit-identical to a splat.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn packets_are_the_rule() {
    use thermite::backend::x86_v3::prelude::*;

    let n = 9u32;
    let ks = f64x4::new([0.0, 1.0, 4.0, 8.0]);
    let (x, w) = ks.gauss_hermite(n);
    let (x, w) = (x.into_array(), w.into_array());
    let alpha = f64x4::new([0.0, 0.5, 2.0, 0.0]);
    let (lx, lw) = ks.gauss_laguerre(alpha, n);
    let (lx, lw) = (lx.into_array(), lw.into_array());
    let ka = ks.into_array();
    let aa = alpha.into_array();
    for k in 0..4 {
        let (sx, sw) = f64x4::splat(ka[k]).gauss_hermite(n);
        assert_eq!(x[k].to_bits(), sx.extract::<0>().to_bits(), "hermite node lane {k}");
        assert_eq!(w[k].to_bits(), sw.extract::<0>().to_bits(), "hermite weight lane {k}");
        let (sx, sw) = f64x4::splat(ka[k]).gauss_laguerre(f64x4::splat(aa[k]), n);
        assert_eq!(lx[k].to_bits(), sx.extract::<0>().to_bits(), "laguerre node lane {k}");
        assert_eq!(lw[k].to_bits(), sw.extract::<0>().to_bits(), "laguerre weight lane {k}");
    }
}
