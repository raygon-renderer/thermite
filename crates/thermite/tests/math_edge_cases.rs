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
// exp-family shoulders: the last binade before overflow and the subnormal
// underflow zone. Each of these was a real defect:
//
// - f64 `exp(709)` returned inf (gate at 708.39; ln(DBL_MAX) is 709.78), and at
//   Best precision the whole subnormal range returned 0
// - f64 `exph(-709)` returned -9.8e307: EXPH's `r - 1` reached -1024, which
//   wraps `pow2n_d`'s biased exponent through the sign bit INSIDE the range gate
// - f64 `exp_m1(-709.5)` returned garbage instead of -1 for the same reason
// - f32 `exp_m1(88.5)` returned NaN at the DEFAULT policy (r = 128 is the NaN
//   exponent field; the answer, 2.7e38, is finite), and `exp_m1(-88.5)` -2.1e38
// - f32 `exph(88.9)` at Medium was NaN: 2^t overflowed before the halving
// -------------------------------------------------------

fn exp_shoulders_f64<S: Simd>(name: &str) {
    type P = Precision;

    let e = |x: f64| Vector::<S::f64x4>::splat(x).exp_p::<P>().extract::<0>();
    let h = |x: f64| Vector::<S::f64x4>::splat(x).exph_p::<P>().extract::<0>();
    let m1 = |x: f64| Vector::<S::f64x4>::splat(x).exp_m1_p::<P>().extract::<0>();

    // Best tier reaches the true domain edges via two-part scaling.
    let got = e(709.78);
    assert!(
        got.is_finite() && (got - 1.7928227943945155e308).abs() < 1e294,
        "[{name}] exp(709.78) should be ~1.79e308, got {got:e}"
    );
    assert!(e(709.79).is_infinite(), "[{name}] exp(709.79) overflows");

    // ...including subnormal results down to the very last one.
    let got = e(-745.0);
    assert!(got > 0.0 && got < 1e-323, "[{name}] exp(-745) should be the min subnormal, got {got:e}");
    let got = h(-709.0);
    assert!(
        (got - 6.083903753117115e-309).abs() < 1e-315,
        "[{name}] exph(-709) should be ~6.08e-309, got {got:e}"
    );

    // exph gets ln 2 more than exp on both sides.
    assert!(h(710.4).is_finite(), "[{name}] exph(710.4) is representable");
    assert!(m1(-709.5) == -1.0, "[{name}] exp_m1(-709.5) saturates to exactly -1");

    let got = Vector::<S::f64x4>::splat(1023.9).exp2_p::<P>().extract::<0>();
    assert!(got.is_finite() && got > 1.6e308, "[{name}] exp2(1023.9) finite, got {got:e}");
    let got = Vector::<S::f64x4>::splat(-1070.0).exp2_p::<P>().extract::<0>();
    assert!(got > 0.0, "[{name}] exp2(-1070) is a subnormal, got {got:e}");
    let got = Vector::<S::f64x4>::splat(308.2).exp10_p::<P>().extract::<0>();
    assert!(got.is_finite() && got > 1.5e308, "[{name}] exp10(308.2) finite, got {got:e}");

    // Average tier: single-scale, but the widened gate and the low-side clamp
    // must hold - sign-garbage was returned inside the old gate.
    let ha = |x: f64| Vector::<S::f64x4>::splat(x).exph_p::<Performance>().extract::<0>();
    let ma = |x: f64| Vector::<S::f64x4>::splat(x).exp_m1_p::<Performance>().extract::<0>();

    assert!(
        Vector::<S::f64x4>::splat(709.0).exp_p::<Performance>().extract::<0>().is_finite(),
        "[{name}] Average exp(709) is representable (gate was 708.39)"
    );
    assert!(ha(710.0).is_finite(), "[{name}] Average exph(710) is representable");
    let got = ha(-709.0);
    assert!(got == 0.0 && got.is_sign_positive(), "[{name}] Average exph(-709) flushes to +0, got {got:e}");
    assert!(ma(-709.5) == -1.0, "[{name}] Average exp_m1(-709.5) is exactly -1, got {:e}", ma(-709.5));
}

fn exp_shoulders_f32<S: Simd>(name: &str) {
    // Best tier (already asymmetric): unchanged contract.
    let got = Vector::<S::f32x8>::splat(89.3f32).exph_p::<Precision>().extract::<0>();
    assert!(got.is_finite() && got > 3.0e38, "[{name}] Best exph(89.3) finite, got {got:e}");

    // Default policy: the NaN and sign-garbage cases.
    let m1 = |x: f32| Vector::<S::f32x8>::splat(x).exp_m1_p::<Performance>().extract::<0>();

    let got = m1(88.5);
    assert!(got.is_infinite() && got > 0.0, "[{name}] Performance exp_m1(88.5) is +inf, NOT NaN; got {got:e}");
    assert!(m1(-88.5) == -1.0, "[{name}] Performance exp_m1(-88.5) is exactly -1, got {:e}", m1(-88.5));

    let got = Vector::<S::f32x8>::splat(-88.0f32).exph_p::<Performance>().extract::<0>();
    assert!(got == 0.0 && got.is_sign_positive(), "[{name}] Performance exph(-88) flushes to +0, got {got:e}");

    // The widened Performance gate: exp(88) is 1.65e38, representable.
    let got = Vector::<S::f32x8>::splat(88.0f32).exp_p::<Performance>().extract::<0>();
    assert!(got.is_finite() && got > 1.6e38, "[{name}] Performance exp(88) finite (gate was 87.3), got {got:e}");

    // Medium tier: the halving now happens in the exponent, so neither end NaNs.
    let hm = |x: f32| Vector::<S::f32x8>::splat(x).exph_p::<MediumP>().extract::<0>();

    let got = hm(88.9);
    assert!(
        got.is_finite() && (got / 2.031188e38 - 1.0).abs() < 1e-2,
        "[{name}] Medium exph(88.9) should be ~2.03e38, got {got:e}"
    );
    let got = hm(-88.5);
    assert!(got == 0.0, "[{name}] Medium exph(-88.5) flushes to 0, got {got:e}");
}

// -------------------------------------------------------
// nth_root at extreme magnitudes. The textbook Halley numerator
// `y * (x - y^N)` is O(x^{(N+1)/N}): for N = 5 it overflowed past x ~ 1e269
// (returning sign-garbage infinities), underflowed below x ~ 1e-250 (silently
// dropping the refinement), and produced NaN at x = 0 and x = inf.
// -------------------------------------------------------

fn nth_root_extremes<S: Simd>(name: &str) {
    fn check5<S: Simd>(name: &str, x: f64, want: f64) {
        let got = Vector::<S::f64x4>::splat(x).nth_root_p::<Precision, 5>().extract::<0>();
        assert!(
            (got - want).abs() <= 1e-12 * want.abs(),
            "[{name}] nth_root5({x:e}): got {got:e}, want {want:e}"
        );
    }

    check5::<S>(name, 1e300, 1e60);
    check5::<S>(name, -1e300, -1e60);
    check5::<S>(name, 1e-300, 1e-60);
    check5::<S>(name, 1e269, 6.309573444801933e53); // the old overflow threshold

    // Degenerate inputs: the dimensionless step's q = y^N/x is 0/0 or inf/inf
    // here, and the guard hands back the (already exact) guess instead.
    let r5 = |x: f64| Vector::<S::f64x4>::splat(x).nth_root_p::<Precision, 5>().extract::<0>();
    assert!(r5(0.0) == 0.0, "[{name}] nth_root5(0) = 0, got {:e}", r5(0.0));
    assert!(r5(f64::INFINITY).is_infinite(), "[{name}] nth_root5(inf) = inf");
    assert!(r5(f64::NEG_INFINITY) == f64::NEG_INFINITY, "[{name}] nth_root5(-inf) = -inf");
    assert!(r5(f64::NAN).is_nan(), "[{name}] nth_root5(NaN) = NaN");

    // N = 4 takes the same generic arm at Best; N = 7 stresses a higher power.
    let got = Vector::<S::f64x4>::splat(1e300).nth_root_p::<Precision, 4>().extract::<0>();
    assert!((got - 1e75).abs() <= 1e-12 * 1e75, "[{name}] nth_root4(1e300), got {got:e}");
    let got = Vector::<S::f64x4>::splat(0.0).nth_root_p::<Precision, 4>().extract::<0>();
    assert!(got == 0.0, "[{name}] nth_root4(0) = 0, got {got:e}");
    let got = Vector::<S::f64x4>::splat(1e-294).nth_root_p::<Precision, 7>().extract::<0>();
    assert!((got - 1e-42).abs() <= 1e-12 * 1e-42, "[{name}] nth_root7(1e-294), got {got:e}");

    // The default policy shares the fixed arm.
    let got = Vector::<S::f64x4>::splat(1e300).nth_root_p::<Performance, 5>().extract::<0>();
    assert!((got - 1e60).abs() <= 1e-9 * 1e60, "[{name}] Performance nth_root5(1e300), got {got:e}");
}

// -------------------------------------------------------
// wrap_angle at large |x|: the old Best path was `x - n * TAU` (fused), which
// drifts by `n * (2pi - TAU)` ~ 0.04 rad by x = 1e15 and returned outright
// WRONG angles (e.g. -3.164 for wrap_angle(1e15 + 1), true value +3.110).
// The Cody-Waite pair carries 2pi to ~110 bits; expected values via mpmath.
// -------------------------------------------------------

fn wrap_angle_large_args<S: Simd>(name: &str) {
    use thermite::vector::ops::MulAddExt;

    type Vd<S> = Vector<<S as Simd>::f64x4>;
    let w = |x: f64| Vector::<S::f64x4>::splat(x).wrap_angle_p::<Precision>().extract::<0>();

    // Valid on every backend: within the exact-product range of the non-FMA
    // split path (|x| <~ 2^29 * 2 pi ~ 3.4e9) and of course the FMA path.
    for (x, want) in [
        (1e8 + 1.0, 2.94269513450401446f64),
        (-1e8 - 1.0, -2.94269513450401446),
        (1e9 + 1.0, 1.5773954235013851694),
        (3e9 + 1.0, 2.7321862705041555082),
    ] {
        let got = w(x);
        assert!(
            (got - want).abs() <= 1e-9,
            "[{name}] wrap_angle({x:e}): got {got}, want {want}"
        );
    }

    // The full range needs single-rounded products, which only FMA hardware
    // provides (emulated FMA is never used in these kernels); the non-FMA split
    // path degrades gradually out here but must stay confined to [-pi, pi).
    if const { <Vd<S> as MulAddExt<Vd<S>, Vd<S>>>::HAS_TRUE_FMA } {
        for (x, want) in [
            (1e15 + 1.0, 3.1096981170701125979f64),
            (5e15 + 1.0, -1.0178800290086099643),
            (1e10 + 1.0, 0.49076892783426521717),
        ] {
            let got = w(x);
            assert!(
                (got - want).abs() <= 1e-9,
                "[{name}] wrap_angle({x:e}): got {got}, want {want}"
            );
        }
    } else {
        for x in [1e10 + 1.0, 1e13, 1e15 + 1.0] {
            let got = w(x);
            assert!(
                (-core::f64::consts::PI..core::f64::consts::PI).contains(&got),
                "[{name}] wrap_angle({x:e}) escaped [-pi, pi): {got}"
            );
        }
    }

    // in range and congruent for f32 too, within the split path's validity
    let got = Vector::<S::f32x8>::splat(3e5f32).wrap_angle_p::<Precision>().extract::<0>();
    let want = 3.03432340346f32; // atan2(sin 3e5, cos 3e5) via mpmath
    assert!(
        (got - want).abs() <= 1e-4,
        "[{name}] wrap_angle_f32(3e5): got {got}, want {want}"
    );
}

// -------------------------------------------------------
// Backend instantiations
// -------------------------------------------------------

macro_rules! suite {
    ($mod_name:ident, $backend:ty, $label:expr) => {
        mod $mod_name {
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

            #[test]
            fn exp_shoulders_f64() {
                super::exp_shoulders_f64::<$backend>($label);
            }

            #[test]
            fn exp_shoulders_f32() {
                super::exp_shoulders_f32::<$backend>($label);
            }

            #[test]
            fn nth_root_extremes() {
                super::nth_root_extremes::<$backend>($label);
            }

            #[test]
            fn wrap_angle_large_args() {
                super::wrap_angle_large_args::<$backend>($label);
            }
        }
    };
}

suite!(scalar, thermite::backend::scalar::Scalar, "scalar");
suite!(x86_v1, thermite::backend::x86_v1::X86V1, "x86_v1");
suite!(x86_v2, thermite::backend::x86_v2::X86V2, "x86_v2");
suite!(x86_v3, thermite::backend::x86_v3::X86V3, "x86_v3");
