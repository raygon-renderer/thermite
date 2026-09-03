//! `ndtr`, `log_ndtr` and `logerfc` against mpmath (`scripts/ndtr_ref.py`, 60 digits).
//!
//! The rows are where the naive spellings fail: the left tail past `ndtr`'s underflow
//! (`x < -38.6`), where only the log form carries anything, and the right side, where
//! `ln(1 - small)` needs `ln_1p`. Several rows sit past `1e15`, where the kernel's
//! `ln_1p(a - 1)` spelling would return `-inf` without its clamp.
//!
//! `ndtr` itself is measured against its own condition number: a rounding in the
//! argument is amplified by `x^2` in the tail (`d ln Phi / d ln x = x^2` there), which is
//! the function's property, not the kernel's. The log forms have no such amplification and
//! are held to a few ulp flat.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]
#![allow(clippy::excessive_precision)]

use thermite::math::policy::DefaultPolicy;
use thermite::math::policy::policies::{AveragePrecision, BestPrecision, MediumPrecision, WorstPrecision};
use thermite::prelude::*;
use thermite_compensated::Compensated;
use thermite_special::{RealSpecialMath, RealSpecialMathWithPolicy, ScalarSpecialMath, SpecialMath};

include!("common/wide.rs");

include!("ndtr_ref/table.rs");

type D = Vector<f64>;
type F = Vector<f32>;

/// Relative error in units of `eps` (absolute where `want` is zero). Matching infinities
/// and matching zeros are exact. A finite/non-finite mismatch is infinite.
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

/// Worst `ulps` over the table for one function, with a per-row condition scale taking
/// `(x, want)`.
fn worst(
    name: &str,
    eps: f64,
    f: impl Fn(f64) -> f64,
    want: impl Fn(&(f64, f64, f64, f64)) -> f64,
    cond: impl Fn(f64, f64) -> f64,
    skip: impl Fn(f64, f64) -> bool,
) -> f64 {
    let mut w = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for row in REFS.iter() {
        let x = row.0;
        let want = want(row);
        if skip(x, want) {
            continue;
        }
        let got = f(x);
        let u = ulps(got, want, eps) / cond(x, want);
        assert!(u.is_finite(), "{name}({x:e}): got {got:e}, want {want:e}");
        if u > w.0 {
            w = (u, x, got, want);
        }
    }
    eprintln!(
        "{name}: worst {:.2} at x = {:e} (got {:e}, want {:e})",
        w.0, w.1, w.2, w.3
    );
    w.0
}

fn no_skip(_: f64, _: f64) -> bool {
    false
}
/// `ndtr`'s tail condition number.
fn tail_cond(x: f64, _: f64) -> f64 {
    1.0 + x * x
}
/// `log_ndtr` on the right is the complement in the log domain and carries `ndtr(-x)`'s
/// condition number. The left arms have none, the log absorbing it.
fn right_cond(x: f64, _: f64) -> f64 {
    if x > 0.0 { 1.0 + x * x } else { 1.0 }
}
/// `logerfc` below `|x| = 1/2` is `ln_1p(+-erf(|x|))`, and below `Best` the f64 `erf`
/// carries an absolute error of about an ulp of 1 near zero (`erf(0)` itself is `2.2e-16`,
/// not `0`), so the result there is accurate absolutely, not relatively. Scaling by
/// `1/|want|` turns the relative measure back into an absolute one, in ulps of 1. At `Best`
/// the kernel takes fdlibm's small-argument arm and the plain relative measure applies.
fn erf_abs_cond(x: f64, want: f64) -> f64 {
    if x.abs() < 0.5 && want != 0.0 {
        1.0 / want.abs()
    } else {
        1.0
    }
}
fn unscaled(_: f64, _: f64) -> f64 {
    1.0
}
/// Below the normal range a result is subnormal or zero, and the default policy flushes
/// subnormals: `ndtr` underflows in the left tail, and the log forms on the right are
/// `-ndtr(-x)`, which underflows the same way. Nothing relative to check there.
fn underflowed_f64(_: f64, want: f64) -> bool {
    want.abs() < f64::MIN_POSITIVE
}
fn underflowed_f32(_: f64, want: f64) -> bool {
    want.abs() < f32::MIN_POSITIVE as f64
}
/// `x^2` leaves binary32 range, or the result is below its normal range.
fn beyond_f32(x: f64, want: f64) -> bool {
    x.abs() > 1e18 || underflowed_f32(x, want)
}

const EPS64: f64 = f64::EPSILON;
const EPS32: f64 = f32::EPSILON as f64;

#[test]
fn f64_ndtr_matches_mpmath_within_its_condition_number() {
    let best = worst(
        "ndtr/best",
        EPS64,
        |x| D::splat(x).ndtr_p::<BestPrecision<DefaultPolicy>>().extract::<0>(),
        |r| r.1,
        tail_cond,
        underflowed_f64,
    );
    let def = worst(
        "ndtr",
        EPS64,
        |x| D::splat(x).ndtr().extract::<0>(),
        |r| r.1,
        tail_cond,
        underflowed_f64,
    );
    assert!(best <= 3.0, "ndtr best: {best} scaled ulp");
    assert!(def <= 4.0, "ndtr default: {def} scaled ulp");
}

#[test]
fn f64_log_ndtr_matches_mpmath() {
    let best = worst(
        "log_ndtr/best",
        EPS64,
        |x| D::splat(x).log_ndtr_p::<BestPrecision<DefaultPolicy>>().extract::<0>(),
        |r| r.2,
        right_cond,
        underflowed_f64,
    );
    let def = worst(
        "log_ndtr",
        EPS64,
        |x| D::splat(x).log_ndtr().extract::<0>(),
        |r| r.2,
        right_cond,
        underflowed_f64,
    );
    assert!(best <= 4.0, "log_ndtr best: {best} ulp");
    assert!(def <= 8.0, "log_ndtr default: {def} ulp");
}

#[test]
fn f64_logerfc_matches_mpmath() {
    let best = worst(
        "logerfc/best",
        EPS64,
        |x| D::splat(x).logerfc_p::<BestPrecision<DefaultPolicy>>().extract::<0>(),
        |r| r.3,
        unscaled,
        no_skip,
    );
    let def = worst(
        "logerfc",
        EPS64,
        |x| D::splat(x).logerfc().extract::<0>(),
        |r| r.3,
        erf_abs_cond,
        no_skip,
    );
    assert!(best <= 4.0, "logerfc best: {best} ulp");
    assert!(def <= 8.0, "logerfc default: {def} ulp");
}

#[test]
fn f64_lower_tiers_stay_on_their_rung() {
    // erfcx's ladder is N = 8/16/24 at Worst/Medium/Average (3.1e-4, 4.3e-7, 4.2e-10
    // normwise), and the logs inherit it.
    let w = worst(
        "log_ndtr/worst",
        1.0,
        |x| D::splat(x).log_ndtr_p::<WorstPrecision<DefaultPolicy>>().extract::<0>(),
        |r| r.2,
        right_cond,
        underflowed_f64,
    );
    let m = worst(
        "log_ndtr/medium",
        1.0,
        |x| {
            D::splat(x)
                .log_ndtr_p::<MediumPrecision<DefaultPolicy>>()
                .extract::<0>()
        },
        |r| r.2,
        right_cond,
        underflowed_f64,
    );
    let a = worst(
        "log_ndtr/average",
        1.0,
        |x| {
            D::splat(x)
                .log_ndtr_p::<AveragePrecision<DefaultPolicy>>()
                .extract::<0>()
        },
        |r| r.2,
        right_cond,
        underflowed_f64,
    );
    assert!(w <= 1e-3, "log_ndtr worst tier: {w:e}");
    assert!(m <= 2e-6, "log_ndtr medium tier: {m:e}");
    assert!(a <= 1e-8, "log_ndtr average tier: {a:e}");

    let w = worst(
        "logerfc/worst",
        1.0,
        |x| D::splat(x).logerfc_p::<WorstPrecision<DefaultPolicy>>().extract::<0>(),
        |r| r.3,
        erf_abs_cond,
        no_skip,
    );
    let m = worst(
        "logerfc/medium",
        1.0,
        |x| D::splat(x).logerfc_p::<MediumPrecision<DefaultPolicy>>().extract::<0>(),
        |r| r.3,
        erf_abs_cond,
        no_skip,
    );
    let a = worst(
        "logerfc/average",
        1.0,
        |x| {
            D::splat(x)
                .logerfc_p::<AveragePrecision<DefaultPolicy>>()
                .extract::<0>()
        },
        |r| r.3,
        erf_abs_cond,
        no_skip,
    );
    assert!(w <= 1e-3, "logerfc worst tier: {w:e}");
    assert!(m <= 2e-6, "logerfc medium tier: {m:e}");
    assert!(a <= 1e-8, "logerfc average tier: {a:e}");
}

#[test]
fn f32_matches_mpmath() {
    // f32 clamps erfcx's ladder at N = 16, about 3.6 f32 ulp.
    let n = worst(
        "f32 ndtr",
        EPS32,
        |x| F::splat(x as f32).ndtr().extract::<0>() as f64,
        |r| r.1,
        tail_cond,
        |x, w| beyond_f32(x, w) || underflowed_f32(x, w),
    );
    let l = worst(
        "f32 log_ndtr",
        EPS32,
        |x| F::splat(x as f32).log_ndtr().extract::<0>() as f64,
        |r| r.2,
        right_cond,
        beyond_f32,
    );
    let e = worst(
        "f32 logerfc",
        EPS32,
        |x| F::splat(x as f32).logerfc().extract::<0>() as f64,
        |r| r.3,
        erf_abs_cond,
        beyond_f32,
    );
    assert!(n <= 4.0, "f32 ndtr: {n} scaled ulp");
    assert!(l <= 8.0, "f32 log_ndtr: {l} ulp");
    assert!(e <= 8.0, "f32 logerfc: {e} ulp");
}

/// The whole reason the log forms exist.
#[test]
fn log_forms_reach_where_the_plain_forms_have_underflowed() {
    assert_eq!(
        D::splat(-40.0).ndtr().extract::<0>(),
        0.0,
        "precondition: ndtr underflows"
    );
    assert_eq!(
        D::splat(30.0).erfc().extract::<0>(),
        0.0,
        "precondition: erfc underflows"
    );

    let want = REFS.iter().find(|r| r.0 == -40.0).unwrap().2;
    assert!(ulps(D::splat(-40.0).log_ndtr().extract::<0>(), want, EPS64) <= 8.0);
    let want = REFS.iter().find(|r| r.0 == 30.0).unwrap().3;
    assert!(ulps(D::splat(30.0).logerfc().extract::<0>(), want, EPS64) <= 8.0);

    // f32 underflows far earlier.
    assert_eq!(
        F::splat(-15.0).ndtr().extract::<0>(),
        0.0,
        "precondition: f32 ndtr underflows"
    );
    let want = REFS.iter().find(|r| r.0 == -20.0).unwrap().2;
    assert!(ulps(F::splat(-20.0).log_ndtr().extract::<0>() as f64, want, EPS32) <= 8.0);
}

#[test]
fn edges() {
    let d = |x: f64| D::splat(x);
    // erfc(0) is 1 to within 2 ulp in this kernel, not exactly 1, and ndtr(0) inherits that.
    assert!(ulps(d(-0.0).ndtr().extract::<0>(), 0.5, EPS64) <= 2.0);
    assert!(ulps(d(0.0).ndtr().extract::<0>(), 0.5, EPS64) <= 2.0);
    assert_eq!(d(f64::NEG_INFINITY).ndtr().extract::<0>(), 0.0);
    assert_eq!(d(f64::INFINITY).ndtr().extract::<0>(), 1.0);
    assert!(d(f64::NAN).ndtr().extract::<0>().is_nan());

    // Both arms agree at zero (the seam) from either side.
    assert_eq!(d(-0.0).log_ndtr().extract::<0>(), d(0.0).log_ndtr().extract::<0>());
    assert!(ulps(d(0.0).log_ndtr().extract::<0>(), -core::f64::consts::LN_2, EPS64) <= 2.0);
    assert_eq!(d(f64::NEG_INFINITY).log_ndtr().extract::<0>(), f64::NEG_INFINITY);
    assert_eq!(d(f64::INFINITY).log_ndtr().extract::<0>(), 0.0);
    assert!(d(f64::NAN).log_ndtr().extract::<0>().is_nan());

    assert_eq!(d(-0.0).logerfc().extract::<0>(), d(0.0).logerfc().extract::<0>());
    // Likewise erf(0) is within 2 ulp of zero rather than zero.
    assert!(d(0.0).logerfc().extract::<0>().abs() <= 2.0 * EPS64);
    assert!(
        ulps(
            d(f64::NEG_INFINITY).logerfc().extract::<0>(),
            core::f64::consts::LN_2,
            EPS64
        ) <= 2.0
    );
    assert_eq!(d(f64::INFINITY).logerfc().extract::<0>(), f64::NEG_INFINITY);
    assert!(d(f64::NAN).logerfc().extract::<0>().is_nan());

    assert!(d(-1e16).log_ndtr().extract::<0>().is_finite());
    assert!(d(-1e100).log_ndtr().extract::<0>().is_finite());
    // The largest x whose square is representable: log_ndtr(-1.3e154) is a finite -8.45e307.
    assert!(d(-1.3e154).log_ndtr().extract::<0>().is_finite());
}

#[test]
fn identities() {
    for &x in &[-8.0, -2.5, -0.7, -0.01, 0.01, 0.7, 2.5, 8.0] {
        let v = D::splat(x);
        // Phi(x) + Phi(-x) = 1.
        let s = v.ndtr().extract::<0>() + (-v).ndtr().extract::<0>();
        assert!(ulps(s, 1.0, EPS64) <= 2.0, "ndtr symmetry at {x}: {s}");
        // The log forms are the logs, where the plain forms exist (bit-for-bit in the
        // moderate region, where they run the same erfc): ln(ndtr(x)) on the left and
        // ln_1p(-ndtr(-x)) on the right (ln(ndtr(8)) itself is the inaccurate spelling, 7%
        // off, which is the point of the function).
        let (ln_ndtr, ln_erfc) = (v.ndtr().extract::<0>().ln(), v.erfc().extract::<0>().ln());
        if (-5.6..=0.0).contains(&x) {
            assert_eq!(
                v.log_ndtr().extract::<0>().to_bits(),
                ln_ndtr.to_bits(),
                "log_ndtr at {x}"
            );
        } else if x > 0.0 {
            let via = (-(-v).ndtr()).ln_1p().extract::<0>();
            assert_eq!(v.log_ndtr().extract::<0>().to_bits(), via.to_bits(), "log_ndtr at {x}");
        } else {
            assert!(
                ulps(v.log_ndtr().extract::<0>(), ln_ndtr, EPS64) <= 4.0,
                "log_ndtr at {x}"
            );
        }
        if (0.5..4.0).contains(&x) {
            assert_eq!(
                v.logerfc().extract::<0>().to_bits(),
                ln_erfc.to_bits(),
                "logerfc at {x}"
            );
        } else {
            // `ln(erfc(x))` is the ill-conditioned spelling wherever erfc is near 1: one ulp
            // of erfc lands as `1/|ln erfc|` ulps of its log, ~89x at |x| = 0.01. The gate
            // carries that factor, so it measures `logerfc` and not the conditioning of what
            // it is compared against. (Measured: 11.8 ulp at x = -0.01 where the scalar
            // `mul_adde` fuses, which is 0.13 ulp of erfc itself.)
            let amp = (1.0f64 / ln_erfc.abs()).max(1.0);
            let got = v.logerfc().extract::<0>();
            assert!(
                ulps(got, ln_erfc, EPS64) <= 4.0 * amp,
                "logerfc at {x}: {got} vs ln(erfc) {ln_erfc}, {} ulp (gate {})",
                ulps(got, ln_erfc, EPS64),
                4.0 * amp
            );
        }
        // logerfc(x) = log_ndtr(-sqrt 2 x) + ln 2. Both sides pass through erf's absolute
        // error near zero, where the identity holds absolutely rather than relatively.
        let via = (v * -D::SQRT_2).log_ndtr().extract::<0>() + core::f64::consts::LN_2;
        let got = v.logerfc().extract::<0>();
        assert!(
            ulps(got, via, EPS64) / erf_abs_cond(x, via) <= 8.0,
            "logerfc via log_ndtr at {x}: {got} vs {via}"
        );
        // probit round trip.
        if x.abs() < 6.0 {
            assert!(
                ulps(v.ndtr().probit().extract::<0>(), x, EPS64) <= 64.0,
                "probit(ndtr({x}))"
            );
        }
    }
}

#[test]
fn scalar_surface() {
    assert!(ulps(0.0f64.scalar_ndtr(), 0.5, EPS64) <= 2.0);
    assert!(ulps(0.0f32.scalar_log_ndtr() as f64, -core::f64::consts::LN_2, EPS32) <= 2.0);
    assert!(0.0f64.scalar_logerfc().abs() <= 2.0 * EPS64);
}

/// `Vector<f64>` is one lane, so everything above exercises one arm per call. A real
/// packet carries every arm at once, and each lane must be bit-identical to a uniform
/// packet of that lane's value on the same backend: the same operations run either way,
/// only the branch skips differ. (A different backend may have different FMA, so the
/// baseline is a splat, not the 1-lane vector.)
#[test]
fn packet_mixes_every_arm_bit_exactly() {
    for xs in [
        [-30.0f64, -0.3, 0.3, 30.0],
        [-8.0, -2.0, 0.7, 6.0],
        [-0.1, -50.0, 1e-3, 3.0],
    ] {
        let v = f64x4::new(xs);
        let l = v.log_ndtr().into_array();
        let e = v.logerfc().into_array();
        let n = v.ndtr().into_array();
        for (k, &x) in xs.iter().enumerate() {
            let s = f64x4::splat(x);
            assert_eq!(
                l[k].to_bits(),
                s.log_ndtr().extract::<0>().to_bits(),
                "log_ndtr lane {k} of {xs:?}"
            );
            assert_eq!(
                e[k].to_bits(),
                s.logerfc().extract::<0>().to_bits(),
                "logerfc lane {k} of {xs:?}"
            );
            assert_eq!(
                n[k].to_bits(),
                s.ndtr().extract::<0>().to_bits(),
                "ndtr lane {k} of {xs:?}"
            );
        }
    }
}

/// `Compensated` inherits the defaults and its direct `erfcx`'s range. Inside it, the
/// values are at least as good as f64's.
#[test]
fn compensated_inherits_the_defaults() {
    for row in REFS.iter().filter(|r| r.0.abs() <= 20.0) {
        let x = Compensated::<D>::new(D::splat(row.0));
        assert!(
            ulps(x.ndtr().value().extract::<0>(), row.1, EPS64) / tail_cond(row.0, row.1) <= 4.0,
            "compensated ndtr at {}",
            row.0
        );
        assert!(
            ulps(x.log_ndtr().value().extract::<0>(), row.2, EPS64) / right_cond(row.0, row.2) <= 8.0,
            "compensated log_ndtr at {}",
            row.0
        );
        assert!(
            ulps(x.logerfc().value().extract::<0>(), row.3, EPS64) / erf_abs_cond(row.0, row.3) <= 8.0,
            "compensated logerfc at {}",
            row.0
        );
    }
}
