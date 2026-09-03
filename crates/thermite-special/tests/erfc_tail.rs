//! `erfc` against a dense mpmath grid (`scripts/erfc_ref.py`), by region, on both the FMA
//! and the non-FMA lowering.
//!
//! The f64 kernel is one product of six rationals times `e^{-x^2}` for every `x`, so the
//! only place its error grows is the tail, where the rounding of `x * x` under the exp is
//! amplified by `x^2`. With a hardware FMA the kernel removes that with the exact residual
//! of the product at every tier. Without one, `Best` removes it with fdlibm's bit-split,
//! and the lower tiers keep the `x^2` growth and are scored against it. The f32 kernel's
//! large-argument arm is a minimax fit documented at 68 ulp, so it is only printed here.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]

use thermite::math::policy::DefaultPolicy;
use thermite::math::policy::policies::BestPrecision;
use thermite::prelude::*;
use thermite_special::{SpecialMath, SpecialMathWithPolicy};

include!("common/wide.rs");

include!("erfc_ref/table.rs");

type Best = BestPrecision<DefaultPolicy>;

const BANDS: [(&str, f64, f64); 5] = [
    ("0..1", 0.0, 1.0),
    ("1..3", 1.0, 3.0),
    ("3..6", 3.0, 6.0),
    ("6..15", 6.0, 15.0),
    ("15..27", 15.0, 27.0),
];

/// Worst relative error in ulps per band, printed, and returned in band order.
fn sweep(name: &str, eps: f64, max_x: f64, f: impl Fn(f64) -> f64) -> [f64; 5] {
    let mut worst = [(0.0f64, 0.0f64); 5];
    for &(x, want) in ERFC.iter() {
        if x > max_x || want < 1e-300 {
            continue;
        }
        let got = f(x);
        let u = ((got - want) / want).abs() / eps;
        assert!(u.is_finite(), "{name}({x}): got {got:e}, want {want:e}");
        for (k, &(_, lo, hi)) in BANDS.iter().enumerate() {
            if x >= lo && x < hi && u > worst[k].0 {
                worst[k] = (u, x);
            }
        }
    }
    let mut out = [0.0; 5];
    for (k, &(band, _, _)) in BANDS.iter().enumerate() {
        eprintln!("{name}: [{band}] worst {:.2} ulp at x = {}", worst[k].0, worst[k].1);
        out[k] = worst[k].0;
    }
    out
}

#[test]
fn f64_scalar_lowering() {
    type D = Vector<f64>;
    let fma = matches!(D::HAS_NATIVE_FMA, thermite::tribool::True);
    eprintln!("Vector<f64> HAS_NATIVE_FMA = {:?}", D::HAS_NATIVE_FMA);
    let def = sweep("f64 1-lane default", f64::EPSILON, 27.0, |x| {
        D::splat(x).erfc().extract::<0>()
    });
    let best = sweep("f64 1-lane best", f64::EPSILON, 27.0, |x| {
        D::splat(x).erfc_p::<Best>().extract::<0>()
    });
    // Mid-range holds a few ulp either way. The tail is flat with FMA at every tier and at
    // `Best` without one. The non-FMA default keeps the x^2 growth (47 ulp at 14, 237 at 24).
    assert!(def[0] <= 6.0 && def[1] <= 6.0, "{def:?}");
    assert!(best.iter().all(|&u| u <= 8.0), "{best:?}");
    if fma {
        assert!(def.iter().all(|&u| u <= 8.0), "{def:?}");
    } else {
        assert!(def[2] <= 20.0 && def[3] <= 100.0 && def[4] <= 400.0, "{def:?}");
    }
}

// x86 and aarch64 only: the assertion below is the point of the test, and wasm's relaxed madd
// reports `Indeterminate`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn f64_wide_fma_lowering() {
    assert!(matches!(f64x4::HAS_NATIVE_FMA, thermite::tribool::True));
    let def = sweep("f64x4 default", f64::EPSILON, 27.0, |x| {
        f64x4::splat(x).erfc().extract::<0>()
    });
    let best = sweep("f64x4 best", f64::EPSILON, 27.0, |x| {
        f64x4::splat(x).erfc_p::<Best>().extract::<0>()
    });
    for w in [def, best] {
        assert!(w.iter().all(|&u| u <= 8.0), "{w:?}");
    }
}

#[test]
fn f32_wide_lowering() {
    let eps = f32::EPSILON as f64;
    let def = sweep("f32x8 default", eps, 9.2, |x| {
        f32x8::splat(x as f32).erfc().extract::<0>() as f64
    });
    let best = sweep("f32x8 best", eps, 9.2, |x| {
        f32x8::splat(x as f32).erfc_p::<Best>().extract::<0>() as f64
    });
    let _ = (def, best);
}

#[test]
fn f32_scalar_lowering() {
    type F = Vector<f32>;
    let eps = f32::EPSILON as f64;
    eprintln!("Vector<f32> HAS_NATIVE_FMA = {:?}", F::HAS_NATIVE_FMA);
    let def = sweep("f32 1-lane default", eps, 9.2, |x| {
        F::splat(x as f32).erfc().extract::<0>() as f64
    });
    let best = sweep("f32 1-lane best", eps, 9.2, |x| {
        F::splat(x as f32).erfc_p::<Best>().extract::<0>() as f64
    });
    let _ = (def, best);
}
