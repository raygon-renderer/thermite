//! The Newton inverses `inv_log_ndtr`, `inv_digamma` and `wright_omega`, against the exact
//! inverses of their f64 arguments (`scripts/inverses_ref.py`, mpmath `findroot` at 50
//! digits from the forwards, never from a seed the kernels use).
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]

use thermite::math::policy::DefaultPolicy;
use thermite::math::policy::policies::{BestPrecision, MediumPrecision, WorstPrecision};
use thermite::prelude::*;
use thermite_special::{RealSpecialMath, RealSpecialMathWithPolicy, SpecialMath};

include!("common/wide.rs");

include!("inverses_ref/table.rs");

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
    if want == 0.0 {
        return got.abs() / eps;
    }
    ((got - want) / want).abs() / eps
}

/// Worst ulps over a `(y, x)` table for `f`, printed. Roots inside the unit interval are
/// scored absolutely, in ulps of 1: `inv_log_ndtr(ln 1/2)` is `2.9e-17` and `dx/dy` is
/// order one there, so a few ulp of `y` is a few `eps` of `x` however small `x` is.
fn worst(name: &str, table: &[(f64, f64)], eps: f64, skip: impl Fn(f64) -> bool, f: impl Fn(f64) -> f64) -> f64 {
    let mut w = (0.0f64, 0.0, 0.0, 0.0);
    for &(y, x) in table {
        if skip(y) {
            continue;
        }
        let got = f(y);
        let u = if x.abs() < 1.0 && x != 0.0 {
            (got - x).abs() / eps
        } else {
            ulps(got, x, eps)
        };
        assert!(u.is_finite(), "{name}({y:e}): got {got:e}, want {x:e}");
        if u > w.0 {
            w = (u, y, got, x);
        }
    }
    eprintln!(
        "{name}: worst {:.2} ulp at y = {:e} (got {:e}, want {:e})",
        w.0, w.1, w.2, w.3
    );
    w.0
}

fn never(_: f64) -> bool {
    false
}

#[test]
fn inv_log_ndtr_f64() {
    let best = worst("inv_log_ndtr best", &INV_LOG_NDTR, f64::EPSILON, never, |y| {
        D::splat(y).inv_log_ndtr_p::<Best>().extract::<0>()
    });
    let def = worst("inv_log_ndtr", &INV_LOG_NDTR, f64::EPSILON, never, |y| {
        D::splat(y).inv_log_ndtr().extract::<0>()
    });
    assert!(best <= 4.0, "{best}");
    assert!(def <= 8.0, "{def}");
    // The lower tiers stop earlier and are scored looser. They must still land.
    let med = worst("inv_log_ndtr medium", &INV_LOG_NDTR, 1.0, never, |y| {
        D::splat(y)
            .inv_log_ndtr_p::<MediumPrecision<DefaultPolicy>>()
            .extract::<0>()
    });
    let wst = worst("inv_log_ndtr worst", &INV_LOG_NDTR, 1.0, never, |y| {
        D::splat(y)
            .inv_log_ndtr_p::<WorstPrecision<DefaultPolicy>>()
            .extract::<0>()
    });
    assert!(med <= 1e-12, "{med:e}");
    assert!(wst <= 1e-9, "{wst:e}");
}

#[test]
fn inv_digamma_f64() {
    let best = worst("inv_digamma best", &INV_DIGAMMA, f64::EPSILON, never, |y| {
        D::splat(y).inv_digamma_p::<Best>().extract::<0>()
    });
    let def = worst("inv_digamma", &INV_DIGAMMA, f64::EPSILON, never, |y| {
        D::splat(y).inv_digamma().extract::<0>()
    });
    assert!(best <= 8.0, "{best}");
    assert!(def <= 16.0, "{def}");
}

#[test]
fn wright_omega_f64() {
    let best = worst("wright_omega best", &WRIGHT_OMEGA, f64::EPSILON, never, |x| {
        D::splat(x).wright_omega_p::<Best>().extract::<0>()
    });
    let def = worst("wright_omega", &WRIGHT_OMEGA, f64::EPSILON, never, |x| {
        D::splat(x).wright_omega().extract::<0>()
    });
    assert!(best <= 4.0, "{best}");
    assert!(def <= 8.0, "{def}");
}

#[test]
fn f32_default() {
    let eps = f32::EPSILON as f64;
    // f32 cannot hold the argument of the deep-tail rows or the x = 1e300 row.
    let in_range = |v: f64| v.abs() > 1e30;
    let a = worst("f32 inv_log_ndtr", &INV_LOG_NDTR, eps, in_range, |y| {
        F::splat(y as f32).inv_log_ndtr().extract::<0>() as f64
    });
    let b = worst(
        "f32 inv_digamma",
        &INV_DIGAMMA,
        eps,
        |y| y > 80.0,
        |y| F::splat(y as f32).inv_digamma().extract::<0>() as f64,
    );
    let c = worst("f32 wright_omega", &WRIGHT_OMEGA, eps, in_range, |x| {
        F::splat(x as f32).wright_omega().extract::<0>() as f64
    });
    // The f32 argument is the f64 row rounded, so the reference is off by the row's own
    // condition number. Only gross failure is asserted here, the table is the f64 gate.
    assert!(a <= 4096.0 && b <= 4096.0 && c <= 4096.0, "{a} {b} {c}");
}

/// Forward of the inverse returns the argument, to the forward's own accuracy times the
/// forward's condition number: on the right `log_ndtr` is a tiny number with a large
/// relative slope (`x phi/Phi / |y|` is 36 at `x = 6`), so a few ulp of `x` is a hundred
/// ulp of `y` there. That is the round trip's property, not the inverse's.
#[test]
fn round_trips() {
    for &(y, _) in INV_LOG_NDTR.iter() {
        let x = D::splat(y).inv_log_ndtr();
        // Fully qualified: importing the specialized trait would make every method below ambiguous.
        let (back, mills) = <D as thermite_special::specialized::SpecializedRealSpecialMath<f64>>::log_ndtr_with_deriv::<
            DefaultPolicy,
        >(x);
        let cond = (x * mills / D::splat(y)).abs().extract::<0>().max(1.0);
        assert!(
            ulps(back.extract::<0>(), y, f64::EPSILON) <= 8.0 * cond,
            "log_ndtr(inv_log_ndtr({y})), cond {cond}"
        );
    }
    for &(y, _) in INV_DIGAMMA.iter() {
        let x = D::splat(y).inv_digamma();
        let back = x.digamma().extract::<0>();
        assert!(
            (back - y).abs() <= 8.0 * f64::EPSILON * y.abs().max(1.0),
            "digamma(inv_digamma({y})) = {back}"
        );
    }
    for &(x, w_ref) in WRIGHT_OMEGA.iter() {
        if w_ref < f64::MIN_POSITIVE {
            continue; // a subnormal w: the test's own `ln` flushes it
        }
        let w = D::splat(x).wright_omega();
        let back = (w + w.ln()).extract::<0>();
        assert!(
            (back - x).abs() <= 8.0 * f64::EPSILON * x.abs().max(1.0),
            "w + ln w at {x} = {back}"
        );
    }
}

#[test]
fn edges() {
    let d = |v: f64| D::splat(v);
    assert_eq!(d(0.0).inv_log_ndtr().extract::<0>(), f64::INFINITY);
    assert_eq!(d(f64::NEG_INFINITY).inv_log_ndtr().extract::<0>(), f64::NEG_INFINITY);
    assert!(d(0.1).inv_log_ndtr().extract::<0>().is_nan());
    assert!(d(f64::NAN).inv_log_ndtr().extract::<0>().is_nan());
    // ln(1/2): the median.
    assert!(d(-core::f64::consts::LN_2).inv_log_ndtr().extract::<0>().abs() <= 4.0 * f64::EPSILON);

    assert_eq!(d(f64::INFINITY).inv_digamma().extract::<0>(), f64::INFINITY);
    assert_eq!(d(f64::NEG_INFINITY).inv_digamma().extract::<0>(), 0.0);
    assert!(d(f64::NAN).inv_digamma().extract::<0>().is_nan());
    // digamma(1) = -gamma.
    assert!(ulps(d(-0.5772156649015329).inv_digamma().extract::<0>(), 1.0, f64::EPSILON) <= 4.0);

    assert_eq!(d(f64::INFINITY).wright_omega().extract::<0>(), f64::INFINITY);
    assert_eq!(d(f64::NEG_INFINITY).wright_omega().extract::<0>(), 0.0);
    assert_eq!(d(-800.0).wright_omega().extract::<0>(), 0.0);
    assert!(d(f64::NAN).wright_omega().extract::<0>().is_nan());
    // omega(1) = 1 exactly.
    assert_eq!(d(1.0).wright_omega().extract::<0>(), 1.0);
}

/// Every arm in one packet, each lane bit-identical to a splat of itself.
#[test]
fn packets_mix_arms_bit_exactly() {
    let ys = [-1e6f64, -700.5, -5.0, -1e-3];
    let got = f64x4::new(ys).inv_log_ndtr().into_array();
    for (k, &y) in ys.iter().enumerate() {
        assert_eq!(
            got[k].to_bits(),
            f64x4::splat(y).inv_log_ndtr().extract::<0>().to_bits(),
            "inv_log_ndtr lane {k}"
        );
    }
    let ys = [-50.0f64, -1.0, 3.0, 30.0];
    let got = f64x4::new(ys).inv_digamma().into_array();
    for (k, &y) in ys.iter().enumerate() {
        assert_eq!(
            got[k].to_bits(),
            f64x4::splat(y).inv_digamma().extract::<0>().to_bits(),
            "inv_digamma lane {k}"
        );
    }
    let xs = [-20.0f64, -4.0, 0.3, 50.0];
    let got = f64x4::new(xs).wright_omega().into_array();
    for (k, &x) in xs.iter().enumerate() {
        assert_eq!(
            got[k].to_bits(),
            f64x4::splat(x).wright_omega().extract::<0>().to_bits(),
            "wright_omega lane {k}"
        );
    }
}
