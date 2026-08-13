//! Edge-case behavior of the math kernels, where the fast paths used to be
//! silently wrong. Each of these was a real defect:
//!
//! - `ln_1p` f32 at Medium precision never formed `1 + x`, so `ln_1p(-0.5)`
//!   returned 177.5 instead of -0.693 - wrong for every input, not just that one
//! - `ldexp` on the flush (non-`Preserve`) path wrapped the exponent at
//!   `i32::MAX`, encoded overflow as NaN rather than infinity, and had an
//!   underflow check that tested the already-clamped exponent and so never fired
//! - large-argument trig: f64 had no Payne-Hanek reduction at any policy, so
//!   `cos_p::<Precision>(1e16)` returned 1.0
//!
//! It also pins the deliberate `<= Average` clamp for huge trig arguments, so
//! that stays a decision rather than drifting into an accident.
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use thermite::math::policy::policies::{MediumPrecision, Performance, Precision};
use thermite::math::policy::{DenormalBehavior, Policy, PolicyParameters, PrecisionPolicy};
use thermite::prelude::*;

type MediumP = MediumPrecision<Performance>;

/// Pinned flush-to-zero policy so the ldexp edge tests are independent of the
/// `preserve_denormals`/`strict_ieee754` features (which flip the default
/// policy's denormal behavior and take the Preserve ldexp path instead).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct FlushPolicy;

impl Policy for FlushPolicy {
    const POLICY: PolicyParameters = PolicyParameters {
        check_overflow: true,
        unroll_loops: true,
        precision: PrecisionPolicy::Average,
        avoid_branching: false,
        max_iterations: 10000,
        use_compensation: false,
        denormal_behavior: DenormalBehavior::FlushToZero,
    };
}

// -------------------------------------------------------
// ln_1p f32, Medium precision
// -------------------------------------------------------

fn ln1p_f32_medium<S: Simd>(name: &str) {
    // Medium tolerance is 10_000 * EPSILON ~= 1.19e-3 relative.
    const TOL: f32 = 1.2e-3;

    let mut cases: Vec<f32> = vec![-0.5, -0.9, -0.99, -0.001, 0.001, 0.5, 1.0, 3.0, 1000.0, 1.0e30];
    // log sweep both sides of zero
    for k in -30..=30 {
        let m = 10.0f32.powi(k / 3);
        cases.push(m);
        if m < 1.0 {
            cases.push(-m);
        }
    }

    for &x in &cases {
        let got = Vector::<S::f32x8>::splat(x).ln_1p_p::<MediumP>().extract::<0>();
        let want = libm::log1pf(x);

        let rel = ((got - want) / want.abs().max(f32::MIN_POSITIVE)).abs();
        assert!(
            rel <= TOL,
            "[{name}] ln_1p_p::<Medium>({x:e}): got {got:e}, want {want:e} (rel err {rel:e})"
        );
    }

    // domain edges (check_overflow is on in Performance)
    let at = |x: f32| Vector::<S::f32x8>::splat(x).ln_1p_p::<MediumP>().extract::<0>();
    assert_eq!(at(-1.0), f32::NEG_INFINITY, "[{name}] ln_1p(-1) should be -inf");
    assert!(at(-2.0).is_nan(), "[{name}] ln_1p(-2) should be NaN");
    assert_eq!(at(f32::INFINITY), f32::INFINITY, "[{name}] ln_1p(inf) should be inf");
    assert!(at(f32::NAN).is_nan(), "[{name}] ln_1p(NaN) should be NaN");
    assert_eq!(at(0.0), 0.0, "[{name}] ln_1p(0) should be 0");
}

// -------------------------------------------------------
// ldexp, default policy (FlushToZero + check_overflow)
// -------------------------------------------------------

fn ldexp_f32_flush_edges<S: Simd>(name: &str) {
    let ldexp = |x: f32, e: i32| -> f32 {
        Vector::<S::f32x8>::splat(x)
            .ldexp_p::<FlushPolicy>(Vector::<S::i32x8>::splat(e))
            .extract::<0>()
    };

    // the reported cases
    assert_eq!(
        ldexp(1.5, -200).to_bits(),
        0.0f32.to_bits(),
        "[{name}] ldexp(1.5, -200)"
    );
    assert_eq!(ldexp(1.0, i32::MAX), f32::INFINITY, "[{name}] ldexp(1.0, i32::MAX)");

    // exponent saturation / overflow encodes inf, not NaN
    assert_eq!(
        ldexp(1.0, i32::MIN).to_bits(),
        0.0f32.to_bits(),
        "[{name}] ldexp(1.0, i32::MIN)"
    );
    assert_eq!(ldexp(1.5, 300), f32::INFINITY, "[{name}] ldexp(1.5, 300)");
    assert_eq!(ldexp(-1.5, 300), f32::NEG_INFINITY, "[{name}] ldexp(-1.5, 300)");
    assert_eq!(ldexp(f32::MAX, 1), f32::INFINITY, "[{name}] ldexp(MAX, 1)");

    // underflow keeps the sign of zero
    assert_eq!(
        ldexp(-1.5, -200).to_bits(),
        (-0.0f32).to_bits(),
        "[{name}] ldexp(-1.5, -200)"
    );

    // in-range values still match libm exactly (skip subnormal expectations:
    // this is the FlushToZero path, subnormal results flush by design)
    for e in [-126, -30, -1, 0, 1, 30, 127] {
        for x in [1.0f32, 1.5, -0.75, core::f32::consts::PI] {
            let want = libm::ldexpf(x, e);
            if !want.is_normal() {
                continue;
            }
            assert_eq!(ldexp(x, e).to_bits(), want.to_bits(), "[{name}] ldexp({x}, {e})");
        }
    }

    // non-finite inputs pass through, for any exponent
    for e in [-1000, 0, 1000, i32::MIN, i32::MAX] {
        assert_eq!(ldexp(f32::INFINITY, e), f32::INFINITY, "[{name}] ldexp(inf, {e})");
        assert_eq!(
            ldexp(f32::NEG_INFINITY, e),
            f32::NEG_INFINITY,
            "[{name}] ldexp(-inf, {e})"
        );
        assert!(ldexp(f32::NAN, e).is_nan(), "[{name}] ldexp(NaN, {e})");
    }

    // signed zeros pass through, for ANY shift - including one that would
    // overflow a normal input (the zero/subnormal case has to win over the
    // overflow case, not the other way around)
    for e in [-1000, -1, 0, 1, 100, 300, 1000, i32::MIN, i32::MAX] {
        assert_eq!(ldexp(0.0, e).to_bits(), 0.0f32.to_bits(), "[{name}] ldexp(0, {e})");
        assert_eq!(ldexp(-0.0, e).to_bits(), (-0.0f32).to_bits(), "[{name}] ldexp(-0, {e})");
    }

    // subnormal input flushes to (signed) zero on this path - documented FTZ
    // semantics - again for any shift, overflowing ones included
    for e in [10, 300, i32::MAX] {
        assert_eq!(
            ldexp(1.0e-40, e).to_bits(),
            0.0f32.to_bits(),
            "[{name}] ldexp(subnormal, {e})"
        );
        assert_eq!(
            ldexp(-1.0e-40, e).to_bits(),
            (-0.0f32).to_bits(),
            "[{name}] ldexp(-subnormal, {e})"
        );
    }
}

fn ldexp_f64_flush_edges<S: Simd>(name: &str) {
    let ldexp = |x: f64, e: i64| -> f64 {
        Vector::<S::f64x4>::splat(x)
            .ldexp_p::<FlushPolicy>(Vector::<S::i64x4>::splat(e))
            .extract::<0>()
    };

    assert_eq!(
        ldexp(1.5, -2000).to_bits(),
        0.0f64.to_bits(),
        "[{name}] ldexp(1.5, -2000)"
    );
    assert_eq!(
        ldexp(-1.5, -2000).to_bits(),
        (-0.0f64).to_bits(),
        "[{name}] ldexp(-1.5, -2000)"
    );
    assert_eq!(ldexp(1.0, i64::MAX), f64::INFINITY, "[{name}] ldexp(1.0, i64::MAX)");
    assert_eq!(
        ldexp(1.0, i64::MIN).to_bits(),
        0.0f64.to_bits(),
        "[{name}] ldexp(1.0, i64::MIN)"
    );
    assert_eq!(ldexp(1.5, 3000), f64::INFINITY, "[{name}] ldexp(1.5, 3000)");
    assert_eq!(ldexp(f64::MAX, 1), f64::INFINITY, "[{name}] ldexp(MAX, 1)");

    for e in [-1022i64, -100, -1, 0, 1, 100, 1023] {
        for x in [1.0f64, 1.5, -0.75, core::f64::consts::PI] {
            let want = libm::ldexp(x, e as i32);
            if !want.is_normal() {
                continue;
            }
            assert_eq!(ldexp(x, e).to_bits(), want.to_bits(), "[{name}] ldexp({x}, {e})");
        }
    }

    for e in [-10000i64, 0, 10000, i64::MIN, i64::MAX] {
        assert_eq!(ldexp(f64::INFINITY, e), f64::INFINITY, "[{name}] ldexp(inf, {e})");
        assert!(ldexp(f64::NAN, e).is_nan(), "[{name}] ldexp(NaN, {e})");
    }
}

// -------------------------------------------------------
// Large-argument trig
// -------------------------------------------------------

fn trig_large_args_best_f64<S: Simd>(name: &str) {
    // Payne-Hanek path: Best+ precision must agree with libm even for huge args.
    const TOL: f64 = 1.0e-12; // absolute; results are O(1)

    let cases: [f64; 9] = [
        1.0e8,
        4.0e9,
        12345678901234.0,
        1.0e15,
        1.0e16,
        1.0e17,
        1.0e100,
        1.0e300,
        f64::MAX,
    ];

    for &x in &cases {
        let v = Vector::<S::f64x4>::splat(x);
        let (s, c) = v.sin_cos_p::<Precision>();
        let (gs, gc) = (s.extract::<0>(), c.extract::<0>());
        let (ws, wc) = (libm::sin(x), libm::cos(x));

        assert!(
            (gs - ws).abs() <= TOL,
            "[{name}] sin_p::<Precision>({x:e}): got {gs}, want {ws}"
        );
        assert!(
            (gc - wc).abs() <= TOL,
            "[{name}] cos_p::<Precision>({x:e}): got {gc}, want {wc}"
        );
    }

    // non-finite still propagates NaN through the Payne-Hanek branch
    for x in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let (s, c) = Vector::<S::f64x4>::splat(x).sin_cos_p::<Precision>();
        assert!(
            s.extract::<0>().is_nan(),
            "[{name}] sin_p::<Precision>({x}) should be NaN"
        );
        assert!(
            c.extract::<0>().is_nan(),
            "[{name}] cos_p::<Precision>({x}) should be NaN"
        );
    }
}

fn trig_large_args_best_f32<S: Simd>(name: &str) {
    const TOL: f32 = 1.0e-6;

    let cases: [f32; 6] = [1.0e5, 1.0e7, 1.0e8, 1.0e16, 1.0e30, f32::MAX];

    for &x in &cases {
        let v = Vector::<S::f32x8>::splat(x);
        let (s, c) = v.sin_cos_p::<Precision>();
        let (gs, gc) = (s.extract::<0>(), c.extract::<0>());
        let (ws, wc) = (libm::sinf(x), libm::cosf(x));

        assert!(
            (gs - ws).abs() <= TOL,
            "[{name}] sinf_p::<Precision>({x:e}): got {gs}, want {ws}"
        );
        assert!(
            (gc - wc).abs() <= TOL,
            "[{name}] cosf_p::<Precision>({x:e}): got {gc}, want {wc}"
        );
    }

    for x in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        let (s, c) = Vector::<S::f32x8>::splat(x).sin_cos_p::<Precision>();
        assert!(
            s.extract::<0>().is_nan(),
            "[{name}] sinf_p::<Precision>({x}) should be NaN"
        );
        assert!(
            c.extract::<0>().is_nan(),
            "[{name}] cosf_p::<Precision>({x}) should be NaN"
        );
    }
}

/// The <= Average tiers deliberately clamp out-of-range trig arguments to zero
/// (sin -> 0, cos -> 1) instead of paying for Payne-Hanek; pin that behavior so
/// a change to it is a deliberate decision, not an accident.
fn trig_large_args_average_clamp<S: Simd>(name: &str) {
    // Pin the Performance (Average) policy explicitly: under `strict_ieee754`
    // the DEFAULT policy is Precision, which correctly uses Payne-Hanek instead.
    let (s, c) = Vector::<S::f64x4>::splat(1.0e16).sin_cos_p::<Performance>();
    assert_eq!(s.extract::<0>(), 0.0, "[{name}] Average-policy sin(1e16) clamp");
    assert_eq!(c.extract::<0>(), 1.0, "[{name}] Average-policy cos(1e16) clamp");

    let (s, c) = Vector::<S::f32x8>::splat(1.0e8f32).sin_cos_p::<Performance>();
    assert_eq!(s.extract::<0>(), 0.0, "[{name}] Average-policy sinf(1e8) clamp");
    assert_eq!(c.extract::<0>(), 1.0, "[{name}] Average-policy cosf(1e8) clamp");
}

// -------------------------------------------------------
// Backend instantiations
// -------------------------------------------------------

macro_rules! suite {
    ($mod_name:ident, $backend:ty, $label:expr) => {
        mod $mod_name {
            use super::*;

            #[test]
            fn ln1p_f32_medium() {
                super::ln1p_f32_medium::<$backend>($label);
            }

            #[test]
            fn ldexp_f32_flush_edges() {
                super::ldexp_f32_flush_edges::<$backend>($label);
            }

            #[test]
            fn ldexp_f64_flush_edges() {
                super::ldexp_f64_flush_edges::<$backend>($label);
            }

            #[test]
            fn trig_large_args_best_f64() {
                super::trig_large_args_best_f64::<$backend>($label);
            }

            #[test]
            fn trig_large_args_best_f32() {
                super::trig_large_args_best_f32::<$backend>($label);
            }

            #[test]
            fn trig_large_args_average_clamp() {
                super::trig_large_args_average_clamp::<$backend>($label);
            }
        }
    };
}

suite!(scalar, thermite::backend::scalar::Scalar, "scalar");
suite!(x86_v1, thermite::backend::x86_v1::X86V1, "x86_v1");
suite!(x86_v2, thermite::backend::x86_v2::X86V2, "x86_v2");
suite!(x86_v3, thermite::backend::x86_v3::X86V3, "x86_v3");
