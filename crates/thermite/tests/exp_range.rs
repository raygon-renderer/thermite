//! `exp` must reach the whole representable range at every tier that checks overflow.
//!
//! A single exponent-field construction of `2^r` caps `r` at the top exponent, and
//! therefore caps the INPUT at 88.3 / 709.42, short of `ln(MAX)` by 0.42 and 0.36. Every
//! result above 2.24e38 / 1.25e308 then comes back `inf` while the true answer is finite,
//! a wrong answer rather than reduced precision. The `<= Average` tiers, which include
//! `DefaultPolicy` on x86, did that until they were moved onto the two-scale
//! reconstruction the accurate tiers already used.
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use thermite::backend::x86_v3::prelude::*;
use thermite::math::policy::DefaultPolicy;
use thermite::math::policy::policies::{Precision, Reference, Size};

/// Largest input whose `exp` is still finite, found by bisection.
fn last_finite_f32(f: impl Fn(f32) -> f32) -> f32 {
    let (mut lo, mut hi) = (0.0f32, 95.0f32);
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        if f(mid).is_finite() && f(mid) != 0.0 {
            lo = mid
        } else {
            hi = mid
        }
    }
    lo
}

fn last_finite_f64(f: impl Fn(f64) -> f64) -> f64 {
    let (mut lo, mut hi) = (0.0f64, 760.0f64);
    for _ in 0..96 {
        let mid = 0.5 * (lo + hi);
        if f(mid).is_finite() && f(mid) != 0.0 {
            lo = mid
        } else {
            hi = mid
        }
    }
    lo
}

/// `ln(FLT_MAX)` and `ln(DBL_MAX)`.
const LN_MAX_F32: f32 = 88.72284;
const LN_MAX_F64: f64 = 709.782712893384;

#[test]
fn exp_reaches_ln_max_at_every_overflow_checking_tier() {
    // Within 0.01 of the true limit. The remaining sliver is the rounded gate constant
    // (88.72 / 709.78), which is shared by every tier here and is not a tier divergence.
    macro_rules! check32 {
        ($name:literal, $p:ty) => {{
            let got = last_finite_f32(|x| f32x8::splat(x).exp_p::<$p>().into_array()[0]);
            assert!(
                (LN_MAX_F32 - got) < 0.01,
                "{} f32: exp finite only to {got}, want ~{LN_MAX_F32} (short by {})",
                $name,
                LN_MAX_F32 - got
            );
        }};
    }
    macro_rules! check64 {
        ($name:literal, $p:ty) => {{
            let got = last_finite_f64(|x| f64x4::splat(x).exp_p::<$p>().into_array()[0]);
            assert!(
                (LN_MAX_F64 - got) < 0.01,
                "{} f64: exp finite only to {got}, want ~{LN_MAX_F64} (short by {})",
                $name,
                LN_MAX_F64 - got
            );
        }};
    }

    check32!("DefaultPolicy", DefaultPolicy);
    check32!("Size", Size);
    check32!("Precision", Precision);
    check32!("Reference", Reference);

    check64!("DefaultPolicy", DefaultPolicy);
    check64!("Size", Size);
    check64!("Precision", Precision);
    check64!("Reference", Reference);
}

/// The specific results that used to come back as infinity.
#[test]
fn the_top_binade_of_exp_is_finite_and_correct() {
    let xs32: [f32; 8] = [88.0, 88.3, 88.4, 88.5, 88.6, 88.7, 88.72, 87.5];
    for (name, got) in [
        ("DefaultPolicy", f32x8::new(xs32).exp().into_array()),
        ("Size", f32x8::new(xs32).exp_p::<Size>().into_array()),
        ("Precision", f32x8::new(xs32).exp_p::<Precision>().into_array()),
    ] {
        for (i, &v) in got.iter().enumerate() {
            let want = libm::expf(xs32[i]);
            assert!(v.is_finite(), "f32 {name}: exp({}) = {v}, want {want}", xs32[i]);
            let rel = ((v as f64 - want as f64) / want as f64).abs();
            assert!(
                rel < 1e-5,
                "f32 {name}: exp({}) = {v}, want {want} (rel {rel})",
                xs32[i]
            );
        }
    }

    let xs64: [f64; 4] = [709.0, 709.4, 709.6, 709.7];
    for (name, got) in [
        ("DefaultPolicy", f64x4::new(xs64).exp().into_array()),
        ("Size", f64x4::new(xs64).exp_p::<Size>().into_array()),
        ("Precision", f64x4::new(xs64).exp_p::<Precision>().into_array()),
    ] {
        for (i, &v) in got.iter().enumerate() {
            let want = libm::exp(xs64[i]);
            assert!(v.is_finite(), "f64 {name}: exp({}) = {v}, want {want}", xs64[i]);
            let rel = ((v - want) / want).abs();
            assert!(
                rel < 1e-12,
                "f64 {name}: exp({}) = {v}, want {want} (rel {rel})",
                xs64[i]
            );
        }
    }
}

/// `compound(x, n) = (1 + x)^n` routes through `exp`, so it inherited the early
/// overflow: `compound(6.77, 43.2)` returned `inf` where the answer is 2.72e38, well
/// inside float32. Kept as its own test because the failure was reported against
/// `compound`, and a future reader should be able to find it under that name.
#[test]
fn compound_reaches_the_top_of_the_range() {
    let x: [f32; 8] = [6.7693954, 2.1459005, 4.1036143, 2.605822, 1.9016389, 5.0, 1.0, 0.5];
    let n: [f32; 8] = [43.166588, 77.04856, 54.38955, 52.417728, 74.708397, 49.0, 127.0, 200.0];

    for (name, got) in [
        ("DefaultPolicy", f32x8::new(x).compound(f32x8::new(n)).into_array()),
        (
            "Precision",
            f32x8::new(x).compound_p::<Precision>(f32x8::new(n)).into_array(),
        ),
    ] {
        for i in 0..8 {
            let want = libm::powf(1.0 + x[i], n[i]);
            if !want.is_finite() {
                continue;
            }
            assert!(
                got[i].is_finite(),
                "{name}: compound({}, {}) = {}, want {want}",
                x[i],
                n[i],
                got[i]
            );
            let rel = ((got[i] as f64 - want as f64) / want as f64).abs();
            assert!(
                rel < 1e-4,
                "{name}: compound({}, {}) = {}, want {want}",
                x[i],
                n[i],
                got[i]
            );
        }
    }
}

/// `powf` with a SUBNORMAL base. Under `preserve_denormals` the base is a real, small,
/// positive number and `x^tiny` is 1.0, but the kernel classified it as zero and applied
/// the `0^negative = inf` rule instead. It also could not reduce a subnormal at all, since
/// there is no exponent field to split.
///
/// **Runs in both denormal modes.** A `#[cfg(feature = ...)]` on the whole test would make
/// it silently vanish from the default build, so the expectation switches instead of the
/// test. Under the default flush, a subnormal base genuinely IS zero by the time the
/// kernel sees it and `0^negative = inf` is correct.
#[test]
fn powf_of_a_subnormal_base() {
    let b: [f32; 8] = [
        1.3754039e-39,
        7.6285623e-39,
        1.1431556e-38,
        1.0e-40,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        5.0e-39,
        9.0e-41,
    ];
    let e: [f32; 8] = [-1.3287e-16, -1.1e-35, -3.6e-25, -1.0e-20, -1.0e-20, -1.0e-10, 2.0, 0.5];

    for (name, got) in [
        ("DefaultPolicy", f32x8::new(b).powf(f32x8::new(e)).into_array()),
        (
            "Precision",
            f32x8::new(b).powf_p::<Precision>(f32x8::new(e)).into_array(),
        ),
    ] {
        for i in 0..8 {
            if cfg!(feature = "preserve_denormals") {
                let want = libm::powf(b[i], e[i]);
                let rel = if want == 0.0 {
                    got[i] as f64
                } else {
                    ((got[i] as f64 - want as f64) / want as f64).abs()
                };
                assert!(
                    got[i].is_finite() && rel < 1e-3,
                    "{name}: powf({}, {}) = {}, want {want} (preserve_denormals)",
                    b[i],
                    e[i],
                    got[i]
                );
            } else {
                // Flushed: the base is zero here, so a negative exponent is a pole and a
                // positive one gives zero. `MIN_POSITIVE` is normal and survives either way.
                let flushed_to_zero = b[i] < f32::MIN_POSITIVE;
                let want = libm::powf(if flushed_to_zero { 0.0 } else { b[i] }, e[i]);
                assert_eq!(
                    got[i].is_infinite(),
                    want.is_infinite(),
                    "{name}: powf({}, {}) = {}, want {want} (denormals flushed)",
                    b[i],
                    e[i],
                    got[i]
                );
            }
        }
    }
}
