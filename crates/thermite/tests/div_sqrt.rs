//! Gate for `approx_div_sqrt`, `$a/\sqrt{b}$`.
//!
//! Three claims to pin:
//!
//! 1. **Every tier is as accurate as its tier promises.** Checked against a binary64
//!    reference for f32, which has enough headroom to resolve a single f32 ulp.
//! 2. **`Best` and above is exactly `self / denom.sqrt()`**, bit for bit. That is the
//!    design rather than an accident (see that test's own comment for the measurement),
//!    so the equality is what guards the path against someone re-adding a
//!    refinement that cannot help.
//! 3. **The edges follow `check_overflow`.** The Newton step is invalid at `b = 0` and
//!    `b = inf`, and the guard that keeps the raw estimate there is gated like every other
//!    edge patch-up in this tree. Both sides of that gate are asserted, so making the
//!    guard unconditional has to be a deliberate change rather than a silent one.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use thermite::math::CoreMathWithPolicy;
use thermite::math::policy::policies::{HighPerformance, Performance, Precision, Reference, Size, UltraPerformance};
use thermite::math::policy::{DefaultPolicy, DenormalBehavior, Policy, PolicyParameters, PrecisionPolicy};

macro_rules! for_each_tier {
    // `$ti` is the tier's index into `TIER_NAMES`, owned by the macro so call sites
    // don't carry a manual counter whose last increment trips `unused_assignments`.
    (|$p:ident, $ti:ident| $body:block) => {{
        let mut $ti = 0;
        {
            type $p = UltraPerformance;
            $body
        }
        {
            $ti += 1;
            type $p = HighPerformance;
            $body
        }
        {
            $ti += 1;
            type $p = Performance;
            $body
        }
        {
            $ti += 1;
            type $p = Size;
            $body
        }
        {
            $ti += 1;
            type $p = DefaultPolicy;
            $body
        }
        {
            $ti += 1;
            type $p = Precision;
            $body
        }
        {
            $ti += 1;
            type $p = Reference;
            $body
        }
    }};
}

const TIER_NAMES: [&str; 7] = [
    "UltraPerformance",
    "HighPerformance",
    "Performance",
    "Size",
    "DefaultPolicy",
    "Precision",
    "Reference",
];

/// Relative tolerance per tier. The first three take the raw or once-refined estimate, and
/// everything from `Size` on is at or near an exact route.
const TOL_F32: [f32; 7] = [1.0e-2, 1.0e-5, 1.0e-5, 1.0e-6, 1.0e-6, 1.0e-6, 1.0e-6];

fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    }
}

/// A hand-built policy that keeps the estimate paths reachable on BOTH sides of the
/// `strict_ieee754` feature: under strict every shipped tier is `Preserve` and takes the
/// exact route, so exercising the strict-gated guards requires asking for flush semantics
/// by name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct FlushAverage;
impl Policy for FlushAverage {
    const POLICY: PolicyParameters = PolicyParameters {
        check_overflow: true,
        unroll_loops: false,
        precision: PrecisionPolicy::Average,
        avoid_branching: false,
        max_iterations: 50,
        use_compensation: false,
        denormal_behavior: DenormalBehavior::FlushToZero,
    };
}

/// Same, at `Worst`: the only precision that reaches `approx_div`'s multiply-by-estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct FlushWorst;
impl Policy for FlushWorst {
    const POLICY: PolicyParameters = PolicyParameters {
        check_overflow: true,
        unroll_loops: false,
        precision: PrecisionPolicy::Worst,
        avoid_branching: false,
        max_iterations: 50,
        use_compensation: false,
        denormal_behavior: DenormalBehavior::FlushToZero,
    };
}

for_each_backend_concrete! {

/// Ordinary arguments, every tier, against a binary64 oracle.
fn matches_a_reference_at_every_tier() {
    let cases: [(f32, f32); 10] = [
        (1.0, 1.0),
        (1.0, 4.0),
        (3.0, 2.0),
        (-3.0, 2.0),
        (0.5, 0.25),
        (1e-8, 3.0),
        (1e8, 7.0),
        (2.0, 1e-20),
        (2.0, 1e20),
        (123.456, 789.012),
    ];

    for_each_tier!(|P, ti| {
        for &(a, b) in &cases {
            let got = f32x8::splat(a).approx_div_sqrt_p::<P>(f32x8::splat(b)).extract::<0>();
            let want = a as f64 / (b as f64).sqrt();

            assert!(
                rel(got as f64, want) <= TOL_F32[ti] as f64,
                "{}: {a}/sqrt({b}) = {got}, want {want}",
                TIER_NAMES[ti]
            );
        }
    });
}

/// At `Best` and above the kernel must be **bit-identical** to `self / denom.sqrt()`.
///
/// This started life as the opposite assertion, that a Karp-Markstein correction beat the
/// naive spelling, and it came back strictly better on 0 of 4096 inputs. A hardware divide
/// already returns the correctly rounded `$a/s$`, so there is nothing there to recover.
/// The equality is the design now: anything clever appearing on that path is either a
/// regression or needs to re-argue the case this test closed.
fn the_best_tier_is_exactly_the_naive_spelling() {
    let mut total = 0;

    // A spread of mantissas rather than round numbers: an exactly-representable quotient
    // has no rounding to correct and would say nothing either way.
    let mut seed = 0x9E3779B9u32;
    for _ in 0..4096 {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let a = f32::from_bits((seed >> 3) | 0x3800_0000) * 3.7;
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let b = f32::from_bits((seed >> 3) | 0x3800_0000) * 1.9;

        if !a.is_finite() || !b.is_finite() || b <= 0.0 {
            continue;
        }

        let kernel = f32x8::splat(a)
            .approx_div_sqrt_p::<Precision>(f32x8::splat(b))
            .extract::<0>();
        let naive = (f32x8::splat(a) / f32x8::splat(b).sqrt()).extract::<0>();

        assert_eq!(
            kernel.to_bits(),
            naive.to_bits(),
            "{a}/sqrt({b}): kernel {kernel} != naive {naive}"
        );

        // And both are within half an ulp of the true value, modulo the half already
        // inside the rounded root, the bound the doc comment claims.
        let want = a as f64 / (b as f64).sqrt();
        assert!(
            rel(kernel as f64, want) <= f32::EPSILON as f64,
            "{a}/sqrt({b}) = {kernel}, want {want}"
        );

        total += 1;
    }

    assert!(total > 3000, "generator produced too few usable cases: {total}");
}

/// The reason `approx_div_sqrt` is used in the real-vector `hypot` kernels even though
/// their scale factor is an exact power of two.
///
/// Scaling by `2^-k` is exact in the normal range, so `s * (1/sqrt(acc))` and
/// `s / sqrt(acc)` agree bit for bit there, at 0 differences in 11940 cases and
/// at 0 on every approximate tier. They stop agreeing once the result underflows, because
/// the multiply then rounds too and the separate form rounds twice. That is not an exotic
/// corner for `inv_hypot`: staying representable where the norm overflows is the entire
/// point of the inverse path, and those answers are subnormal.
///
/// So this asserts the shape of the disagreement rather than a tolerance: identical in the
/// normal range, and the fused form winning the subnormal tail on balance.
fn the_two_spellings_diverge_only_in_the_subnormal_tail() {
    let (mut normal_diff, mut sub_fused_better, mut sub_split_better) = (0, 0, 0);

    for ki in 0..60i32 {
        for m in 1..200u32 {
            let acc = 1.0f32 + (m as f32) / 100.0;

            for (k, subnormal) in [(ki - 30, false), (-(ki + 68), true)] {
                let s = f32::powi(2.0, k);
                if !s.is_normal() {
                    continue;
                }

                let split = (f32x8::splat(s) * f32x8::splat(acc).inverse_sqrt_p::<Precision>()).extract::<0>();
                let fused = f32x8::splat(s)
                    .approx_div_sqrt_p::<Precision>(f32x8::splat(acc))
                    .extract::<0>();

                let want = s as f64 / (acc as f64).sqrt();
                if want == 0.0 || !want.is_finite() {
                    continue;
                }

                if split.to_bits() == fused.to_bits() {
                    continue;
                }

                if !subnormal {
                    normal_diff += 1;
                    continue;
                }

                let e_split = rel(split as f64, want);
                let e_fused = rel(fused as f64, want);
                if e_fused < e_split {
                    sub_fused_better += 1;
                } else if e_split < e_fused {
                    sub_split_better += 1;
                }
            }
        }
    }

    assert_eq!(
        normal_diff, 0,
        "the two spellings must be bit-identical wherever the result is normal"
    );

    // Double rounding lands closer by luck occasionally, so this is a skew rather than a
    // sweep. Measured 48 to 10, and a reversal means the fused form stopped
    // being the better one and the kernels should go back.
    assert!(
        sub_fused_better > 2 * sub_split_better,
        "subnormal tail: fused better on {sub_fused_better}, split better on {sub_split_better}"
    );
}

/// `inverse_sqrt` carries the identical guard, and is tested here because it is the same
/// defect and the same fix rather than a neighbouring one.
///
/// The Newton step in `inverse_sqrt_internal` returned NaN for both
/// `x = 0` and `x = inf` at every refining tier, `Performance` included, so unlike the
/// `approx_div_sqrt` case this was not a tier opting out of `check_overflow`, it was
/// unguarded everywhere. `rsqrt` had already produced `+inf` and `0` correctly, and the
/// refinement was what destroyed them.
fn inverse_sqrt_edges_follow_the_overflow_policy() {
    for_each_tier!(|P, ti| {
        let f = |x: f32| f32x8::splat(x).inverse_sqrt_p::<P>().extract::<0>();
        let refines = const { <P as Policy>::POLICY.precision.gt(PrecisionPolicy::Worst) };
        let guarded = const { <P as Policy>::POLICY.check_overflow };
        // The `Best` and `Preserve` escapes take the exact route before any refinement
        // can run. Under `strict_ieee754` every shipped tier is `Preserve`, so all of
        // them land here and get the IEEE edges regardless of the guard.
        let exact = const {
            <P as Policy>::POLICY.precision.ge(PrecisionPolicy::Best)
                || matches!(<P as Policy>::POLICY.denormal_behavior, DenormalBehavior::Preserve)
                || !f32x8::HAS_APPROX_RSQRT
        };

        // The patch is applied whenever `check_overflow` asks, at every tier on the
        // estimate. Without it the answer is whatever the backend's raw estimate and the
        // unguarded step give: `+inf` on x86 at the raw tier, NaN once refined, NaN on
        // NEON even raw (`rsqrt` itself stays raw by design). Only garbage is ruled out.
        if exact || guarded {
            assert!(
                f(0.0).is_infinite() && f(0.0) > 0.0,
                "{}: 1/sqrt(0) = {}",
                TIER_NAMES[ti],
                f(0.0)
            );
            assert_eq!(
                f(f32::INFINITY),
                0.0,
                "{}: 1/sqrt(inf) = {}",
                TIER_NAMES[ti],
                f(f32::INFINITY)
            );
        } else {
            let _ = refines;
            assert!(
                f(0.0).is_nan() || (f(0.0).is_infinite() && f(0.0) > 0.0),
                "{}: unguarded 1/sqrt(0) = {}, want NaN or +inf",
                TIER_NAMES[ti],
                f(0.0)
            );
        }

        // The interior is untouched by the guard.
        assert!(
            (f(4.0) - 0.5).abs() < 1e-3,
            "{}: 1/sqrt(4) = {}",
            TIER_NAMES[ti],
            f(4.0)
        );
        assert!(f(-1.0).is_nan(), "{}: 1/sqrt(-1) should be NaN", TIER_NAMES[ti]);
    });
}

/// f64 as well. Where the backend has no approximate f64 rsqrt (x86 below AVX-512, wasm,
/// scalar) every tier takes an exact route and the two spellings agree closely
/// throughout. Where it does (AVX-512's 14-bit `vrsqrt14pd`, NEON) the fast tiers keep
/// the estimate and get the same per-tier budget as f32.
fn f64_is_accurate_at_every_tier() {
    let cases: [(f64, f64); 6] = [
        (1.0, 2.0),
        (3.0, 7.0),
        (-5.0, 11.0),
        (1e-100, 3.0),
        (1e100, 3.0),
        (2.5, 1e-300),
    ];

    for_each_tier!(|P, ti| {
        let exact = const {
            <P as Policy>::POLICY.precision.ge(PrecisionPolicy::Best)
                || matches!(<P as Policy>::POLICY.denormal_behavior, DenormalBehavior::Preserve)
        };
        let tol = if exact || !f64x4::HAS_APPROX_RSQRT { 1e-15 } else { TOL_F32[ti] as f64 };

        for &(a, b) in &cases {
            let got = f64x4::splat(a).approx_div_sqrt_p::<P>(f64x4::splat(b)).extract::<0>();
            let naive = a / b.sqrt();

            assert!(
                rel(got, naive) <= tol,
                "{}: {a}/sqrt({b}) = {got}, naive gives {naive}",
                TIER_NAMES[ti]
            );
        }
    });
}

/// The edges, which are inherited from the root and the divide rather than invented here.
///
/// `b = 0` and `b = inf` are the two points where the Newton step is invalid: the estimate
/// is `+inf` and `0` respectively, and refining either evaluates `inf * NaN`. The kernel
/// keeps the raw estimate there instead, but only under `check_overflow`, like every other
/// edge patch-up in this tree, because the guard is four operations on top of a four
/// operation refinement and the tiers that opt out are the ones paying for speed.
///
/// So this asserts the real behavior on both sides of that gate rather than pretending it
/// is uniform. Tiers without `check_overflow` land in the second branch, where the
/// answer is whatever the raw estimate or the unguarded step gives (x86 raw: `+inf`,
/// refined: NaN. NEON: NaN even raw, since its register `rsqrt` carries a step).
fn edge_cases_follow_the_overflow_policy() {
    for_each_tier!(|P, ti| {
        let f = |a: f32, b: f32| f32x8::splat(a).approx_div_sqrt_p::<P>(f32x8::splat(b)).extract::<0>();
        let refines = const { <P as Policy>::POLICY.precision.gt(PrecisionPolicy::Worst) };
        let guarded = const { <P as Policy>::POLICY.check_overflow };
        // Same as `inverse_sqrt` above: the `Best`/`Preserve` escapes (all shipped tiers
        // under `strict_ieee754`) reach the exact route and its IEEE edges first.
        let exact = const {
            <P as Policy>::POLICY.precision.ge(PrecisionPolicy::Best)
                || matches!(<P as Policy>::POLICY.denormal_behavior, DenormalBehavior::Preserve)
                || !f32x8::HAS_APPROX_RSQRT
        };

        // sqrt of a negative is NaN, and it propagates at every tier.
        assert!(f(1.0, -1.0).is_nan(), "{}", TIER_NAMES[ti]);

        // 0/sqrt(x) is a signed zero at every tier, an ordinary interior point.
        assert_eq!(f(0.0, 4.0), 0.0, "{}", TIER_NAMES[ti]);

        if exact || guarded {
            // a/sqrt(0) is a signed infinity, and the sign of the numerator carries.
            assert!(f(1.0, 0.0).is_infinite() && f(1.0, 0.0) > 0.0, "{}", TIER_NAMES[ti]);
            assert!(f(-1.0, 0.0).is_infinite() && f(-1.0, 0.0) < 0.0, "{}", TIER_NAMES[ti]);

            // a/sqrt(inf) is a signed zero.
            assert_eq!(f(1.0, f32::INFINITY), 0.0, "{}", TIER_NAMES[ti]);
        } else {
            // Unpatched: the raw estimate (x86 `+inf`, NEON NaN) or the unguarded
            // refinement (NaN). Documented, not accidental. Only garbage is ruled out.
            let _ = refines;
            let got = f(1.0, 0.0);
            assert!(
                got.is_nan() || (got.is_infinite() && got > 0.0),
                "{}: unpatched 1/sqrt(0) = {got}, want NaN or +inf",
                TIER_NAMES[ti]
            );
        }

        // A quotient large enough to overflow saturates to infinity rather than wrapping
        // or NaN-ing, on every tier, since nothing in the refinement touches this.
        let over = f(f32::MAX, 1e-30);
        assert!(
            over.is_infinite() && over > 0.0,
            "{}: overflow gave {over}",
            TIER_NAMES[ti]
        );
    });
}

/// `approx_reciprocal`'s Newton step is invalid at `x = 0` and `x = inf`, where the estimate
/// is already exactly right at both (`+-inf` and `+-0`), and refining it evaluates
/// `inf * NaN`. That NaN is deliberate on the fast tiers: no guard is spent on it. Under
/// `strict_ieee754` the edges come out IEEE-correct anyway, because the registers set
/// `HAS_APPROX_RCP = false` under the feature and `rcp()` IS the exact divide. The fix
/// lives at the backend layer, not in the kernel. Both sides asserted: the strict arm is
/// what breaks if a backend ever keeps its estimate under strict, and the fast arm is
/// what breaks if someone re-adds a guard.
fn approx_reciprocal_edges_follow_strict_ieee754() {
    let f = |x: f32| f32x8::splat(x).approx_reciprocal_p::<FlushAverage>().extract::<0>();

    // The interior refines normally either way.
    assert!((f(4.0) - 0.25).abs() < 1e-5, "1/4 = {}", f(4.0));
    assert!((f(-3.0) + 1.0 / 3.0).abs() < 1e-5, "1/-3 = {}", f(-3.0));

    if cfg!(feature = "strict_ieee754") || !f32x8::HAS_APPROX_RCP {
        assert!(f(0.0).is_infinite() && f(0.0) > 0.0, "strict: 1/0 = {}", f(0.0));
        assert!(f(-0.0).is_infinite() && f(-0.0) < 0.0, "strict: 1/-0 = {}", f(-0.0));
        assert_eq!(f(f32::INFINITY), 0.0, "strict: 1/inf");
    } else {
        assert!(f(0.0).is_nan(), "unguarded refinement gives NaN at 0, got {}", f(0.0));
        assert!(f(f32::INFINITY).is_nan(), "unguarded refinement gives NaN at inf");
    }
}

/// `approx_div`'s estimate path treats a DENORMAL divisor as zero (`rcp` does so in
/// hardware regardless of MXCSR), so `0 / denormal` manufactures `0 * inf = NaN` and a
/// nonzero numerator a wrong infinity whose correct answer is finite. No select can
/// produce that finite quotient, so under `strict_ieee754` the whole estimate path is
/// compiled out in favor of the real divide, asserted on both sides here.
fn approx_div_denormal_divisor_follows_strict_ieee754() {
    let f = |a: f32, b: f32| {
        f32x8::splat(a)
            .approx_div_p::<FlushWorst>(f32x8::splat(b))
            .extract::<0>()
    };

    // A normal divisor is fine on both sides (to estimate accuracy, ~12 bits).
    assert!((f(1.0, 4.0) - 0.25).abs() < 1e-3, "1/4 = {}", f(1.0, 4.0));

    let denormal = 1e-39f32;
    assert!(
        denormal != 0.0 && !denormal.is_normal(),
        "test constant must be subnormal"
    );

    // Not gated on `HAS_APPROX_RCP`: an exact-rcp backend under a flushing policy
    // still flushes the divisor to zero and lands in the second branch.
    if cfg!(feature = "strict_ieee754") {
        // The real divide: IEEE everywhere, including the denormal divisor.
        assert_eq!(f(0.0, denormal), 0.0, "strict: 0/denormal");
        let q = f(1e-3, denormal);
        assert!(
            q.is_finite() && (q / 1e36 - 1.0).abs() < 1e-6,
            "strict: 1e-3/1e-39 = {q}"
        );
    } else {
        // The estimate saw zero: documented, not accidental.
        assert!(
            f(0.0, denormal).is_nan(),
            "0 * inf estimate gives NaN, got {}",
            f(0.0, denormal)
        );
        assert!(
            f(1e-3, denormal).is_infinite(),
            "estimate gives inf, got {}",
            f(1e-3, denormal)
        );
    }
}

}
