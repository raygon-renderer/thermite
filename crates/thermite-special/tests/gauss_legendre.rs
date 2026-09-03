//! Gauss-Legendre nodes and weights against mpmath roots (`scripts/bessel_ratio_ref.py`,
//! `findroot` on `P_n` at 50 digits), one root per lane.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]

use thermite::math::policy::DefaultPolicy;
use thermite::math::policy::policies::BestPrecision;
use thermite::prelude::*;
use thermite_special::{RealSpecialMath, RealSpecialMathWithPolicy};

include!("common/wide.rs");

include!("bessel_ratio_ref/table.rs");

type D = Vector<f64>;
type F = Vector<f32>;

#[test]
fn nodes_and_weights_f64() {
    let mut wx = (0.0f64, 0, 0);
    let mut ww = (0.0f64, 0, 0);
    for &(n, k, x, w) in GAUSS_LEGENDRE.iter() {
        for (name, (gx, gw)) in [
            ("default", D::splat(k as f64).gauss_legendre(n)),
            (
                "best",
                D::splat(k as f64).gauss_legendre_p::<BestPrecision<DefaultPolicy>>(n),
            ),
        ] {
            // Nodes absolutely (they live in [-1, 1] and the rule's error is absolute).
            // Weights relatively, scaled by their condition number in the node: at a root
            // the Legendre ODE gives P''/P' = 2x/(1 - x^2), so d ln w / dx = -6x/(1 - x^2),
            // which is 900 at the extreme root of P_33: a quarter ulp of node there is two
            // hundred ulp of weight, and the rule is still exact (see the monomial test).
            let ex = (gx.extract::<0>() - x).abs() / f64::EPSILON;
            let cond = 1.0 + 6.0 * x.abs() / ((1.0 - x) * (1.0 + x));
            let ew = ((gw.extract::<0>() - w) / w).abs() / f64::EPSILON / cond;
            assert!(
                ex <= 8.0,
                "{name} node n={n} k={k}: {} vs {x}, {ex:.1} eps",
                gx.extract::<0>()
            );
            assert!(
                ew <= 8.0,
                "{name} weight n={n} k={k}: {} vs {w}, {ew:.1} scaled ulp",
                gw.extract::<0>()
            );
            if ex > wx.0 {
                wx = (ex, n, k);
            }
            if ew > ww.0 {
                ww = (ew, n, k);
            }
        }
    }
    eprintln!(
        "nodes worst {:.2} eps at n={} k={}; weights worst {:.2} scaled ulp at n={} k={}",
        wx.0, wx.1, wx.2, ww.0, ww.1, ww.2
    );
}

/// The rule integrates x^m exactly for m < 2n: sum w x^m = 2/(m+1) for even m, 0 for odd.
#[test]
fn integrates_monomials_exactly() {
    for n in [1u32, 2, 3, 5, 8, 16, 40] {
        let mut sum = vec![0.0f64; 2 * n as usize];
        for k in 0..n {
            let (x, w) = D::splat(k as f64).gauss_legendre(n);
            let (x, w) = (x.extract::<0>(), w.extract::<0>());
            let mut xm = 1.0;
            for s in sum.iter_mut() {
                *s += w * xm;
                xm *= x;
            }
        }
        for (m, s) in sum.iter().enumerate() {
            let want = if m % 2 == 0 { 2.0 / (m as f64 + 1.0) } else { 0.0 };
            assert!(
                (s - want).abs() <= 32.0 * f64::EPSILON * n as f64,
                "n={n} m={m}: {s} vs {want}"
            );
        }
    }
}

#[test]
fn symmetry_and_edges() {
    for n in [2u32, 7, 16] {
        for k in 0..n {
            let (x, w) = D::splat(k as f64).gauss_legendre(n);
            let (x2, w2) = D::splat((n - 1 - k) as f64).gauss_legendre(n);
            assert!(
                (x.extract::<0>() + x2.extract::<0>()).abs() <= 4.0 * f64::EPSILON,
                "n={n} k={k} mirror"
            );
            assert!(
                ((w.extract::<0>() - w2.extract::<0>()) / w.extract::<0>()).abs() <= 8.0 * f64::EPSILON,
                "n={n} k={k} weight mirror"
            );
        }
    }
    let (x, w) = D::splat(0.0).gauss_legendre(1);
    assert_eq!((x.extract::<0>(), w.extract::<0>()), (0.0, 2.0));
    let (x, w) = D::splat(5.0).gauss_legendre(5);
    assert!(x.extract::<0>().is_nan() && w.extract::<0>().is_nan());
    let (x, _) = D::splat(1.5).gauss_legendre(5);
    assert!(x.extract::<0>().is_nan());
}

#[test]
fn f32_default() {
    let mut worst = 0.0f64;
    for &(n, k, x, w) in GAUSS_LEGENDRE.iter() {
        if n > 64 {
            continue;
        }
        let (gx, gw) = F::splat(k as f32).gauss_legendre(n);
        let ex = (gx.extract::<0>() as f64 - x).abs() / f32::EPSILON as f64;
        let cond = 1.0 + 6.0 * x.abs() / ((1.0 - x) * (1.0 + x));
        let ew = ((gw.extract::<0>() as f64 - w) / w).abs() / f32::EPSILON as f64 / cond;
        worst = worst.max(ex).max(ew);
        assert!(
            ex <= 8.0 && ew <= 8.0,
            "f32 n={n} k={k}: node {ex:.1} eps, weight {ew:.1} scaled ulp"
        );
    }
    eprintln!("f32 worst {worst:.2}");
}

/// A packet of consecutive indices is the rule: every lane bit-identical to a splat.
#[test]
fn packet_is_the_rule() {
    let n = 11u32;
    let ks = f64x4::new([0.0, 1.0, 2.0, 3.0]);
    let (x, w) = ks.gauss_legendre(n);
    let (x, w) = (x.into_array(), w.into_array());
    for k in 0..4 {
        let (sx, sw) = f64x4::splat(k as f64).gauss_legendre(n);
        assert_eq!(x[k].to_bits(), sx.extract::<0>().to_bits(), "node lane {k}");
        assert_eq!(w[k].to_bits(), sw.extract::<0>().to_bits(), "weight lane {k}");
    }
}
