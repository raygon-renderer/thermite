//! Correctness gate for `thermite-special` (which had **zero** tests).
//!
//! Each special function is swept over its valid domain and compared against
//! `libm` with a loose relative tolerance, the same philosophy as the core
//! `diff_math` gate: catch structural bugs (wrong sign, wrong identity, NaN
//! for a finite result, backend divergence), not audit ULPs. The default
//! `Performance` policy is in effect, so the bound is deliberately generous.
include!("common/wide.rs");

use thermite::math::policy::DefaultPolicy;
use thermite::math::policy::policies::Precision;
use thermite_special::{
    RealSpecialMath as _, RealSpecialMathWithPolicy as _, SpecialMath as _, SpecialMathWithPolicy as _,
};

fn close(got: f64, want: f64, tol: f64) -> bool {
    if got.is_nan() || want.is_nan() {
        return got.is_nan() == want.is_nan();
    }
    if got == want {
        return true;
    }
    if !got.is_finite() || !want.is_finite() {
        return got == want;
    }
    (got - want).abs() <= tol * want.abs().max(1.0)
}

/// Sweep `f` over `n` points of `[lo, hi]`, comparing each lane to `oracle`.
macro_rules! sweep_f32 {
    ($name:literal, $method:ident, $oracle:expr, $lo:expr, $hi:expr, $tol:expr) => {{
        let n = 4000usize;
        let mut buf = [0.0f32; 8];
        let mut i = 0;
        while i < n {
            for k in 0..8 {
                let t = ((i + k) as f32) / (n as f32);
                buf[k] = $lo + t * ($hi - $lo);
            }
            let got = f32x8::new(buf).$method().into_array();
            for k in 0..8 {
                let want = $oracle(buf[k]);
                assert!(
                    close(got[k] as f64, want as f64, $tol),
                    "{} f32 @ x={}: got {}, libm {}",
                    $name,
                    buf[k],
                    got[k],
                    want
                );
            }
            i += 8;
        }
    }};
}

macro_rules! sweep_f64 {
    ($name:literal, $method:ident, $oracle:expr, $lo:expr, $hi:expr, $tol:expr) => {{
        let n = 4000usize;
        let mut buf = [0.0f64; 4];
        let mut i = 0;
        while i < n {
            for k in 0..4 {
                let t = ((i + k) as f64) / (n as f64);
                buf[k] = $lo + t * ($hi - $lo);
            }
            let got = f64x4::new(buf).$method().into_array();
            for k in 0..4 {
                let want = $oracle(buf[k]);
                assert!(
                    close(got[k], want, $tol),
                    "{} f64 @ x={}: got {}, libm {}",
                    $name,
                    buf[k],
                    got[k],
                    want
                );
            }
            i += 4;
        }
    }};
}

#[test]
fn erf_f32() {
    sweep_f32!("erf", erf, libm::erff, -6.0, 6.0, 2.0e-3);
}
#[test]
fn erf_f64() {
    sweep_f64!("erf", erf, libm::erf, -6.0, 6.0, 1.0e-6);
}

#[test]
fn erfc_f32() {
    sweep_f32!("erfc", erfc, libm::erfcf, -4.0, 4.0, 3.0e-3);
}
#[test]
fn erfc_f64() {
    sweep_f64!("erfc", erfc, libm::erfc, -4.0, 4.0, 1.0e-6);
}

#[test]
fn tgamma_f32() {
    // Stay on the positive, non-pole side away from the rapid growth tail.
    sweep_f32!("tgamma", tgamma, libm::tgammaf, 0.1, 8.0, 5.0e-3);
}
#[test]
fn tgamma_f64() {
    sweep_f64!("tgamma", tgamma, libm::tgamma, 0.1, 8.0, 1.0e-6);
}

// The large-argument branch of `tgamma`, which no other test reaches.
//
// `tgamma_impl` splits `zgh^(z - 1/2)` into a half-exponent `h` so the full power
// never forms, then folds the tiny `denom` in BETWEEN the two halves. Associating
// it as `(h * h) * denom` instead overflows to +inf from x = 142.75 (f64) and
// x = 27.25 (f32) upward, the whole top third of the finite domain, and that
// shipped, because every sweep above stops at x = 8.
//
// Both tiers are checked on purpose. `Precision` has the exact factorial loop, so
// it is correct at integer arguments even when the split is broken, so only the
// non-integers expose it. `DefaultPolicy` has no such loop and fails everywhere.
// Testing the top tier alone would have caught three quarters of the bug.
#[test]
fn tgamma_f64_large_argument() {
    // 171.61 is the last finite Gamma in f64.
    sweep_f64!("tgamma", tgamma, libm::tgamma, 140.0, 171.6, 1.0e-6);

    // `sweep_f64!` takes an `ident`, so the policy form is spelled out.
    let n = 4000usize;
    let mut buf = [0.0f64; 4];
    let mut i = 0;
    while i < n {
        for k in 0..4 {
            let t = ((i + k) as f64) / (n as f64);
            buf[k] = 140.0 + t * 31.6;
        }
        let got = f64x4::new(buf).tgamma_p::<Precision>().into_array();
        for k in 0..4 {
            let want = libm::tgamma(buf[k]);
            assert!(
                close(got[k], want, 1.0e-9),
                "tgamma f64 Precision @ x={}: got {}, libm {}",
                buf[k],
                got[k],
                want
            );
        }
        i += 4;
    }
}

#[test]
fn tgamma_f32_large_argument() {
    // 34.65 is the last finite Gamma in f32.
    sweep_f32!("tgamma", tgamma, libm::tgammaf, 25.0, 34.6, 3.0e-3);

    let n = 4000usize;
    let mut buf = [0.0f32; 8];
    let mut i = 0;
    while i < n {
        for k in 0..8 {
            let t = ((i + k) as f32) / (n as f32);
            buf[k] = 25.0 + t * 9.6;
        }
        let got = f32x8::new(buf).tgamma_p::<Precision>().into_array();
        for k in 0..8 {
            let want = libm::tgammaf(buf[k]);
            assert!(
                close(got[k] as f64, want as f64, 3.0e-3),
                "tgamma f32 Precision @ x={}: got {}, libm {}",
                buf[k],
                got[k],
                want
            );
        }
        i += 8;
    }
}

// Past the overflow point the answer IS infinity, and this is not something the
// arithmetic can reach on its own: `h` overflows while `denom` underflows, and
// `inf * 0` is NaN. Before the guard, `tgamma(1e30)` and `tgamma(inf)` both returned
// NaN, as did everything above ~300 (f64) and ~100 (f32).
//
// `DefaultPolicy` and `Precision` both check overflow, so both must produce it.
#[test]
fn tgamma_overflows_to_infinity() {
    let big64 = [180.0f64, 300.0, 1.0e10, 1.0e30, 1.0e300, f64::INFINITY];
    for chunk in big64.chunks(4) {
        let mut buf = [200.0f64; 4];
        buf[..chunk.len()].copy_from_slice(chunk);

        for (name, got) in [
            ("DefaultPolicy", f64x4::new(buf).tgamma().into_array()),
            ("Precision", f64x4::new(buf).tgamma_p::<Precision>().into_array()),
        ] {
            for (i, v) in got.iter().enumerate() {
                assert!(
                    v.is_infinite() && v.is_sign_positive(),
                    "f64 {name}: tgamma({}) = {v}, want +inf",
                    buf[i]
                );
            }
        }
    }

    let big32 = [40.0f32, 100.0, 1.0e10, 1.0e30, 3.0e38, f32::INFINITY, 60.0, 500.0];
    for (name, got) in [
        ("DefaultPolicy", f32x8::new(big32).tgamma().into_array()),
        ("Precision", f32x8::new(big32).tgamma_p::<Precision>().into_array()),
    ] {
        for (i, v) in got.iter().enumerate() {
            assert!(
                v.is_infinite() && v.is_sign_positive(),
                "f32 {name}: tgamma({}) = {v}, want +inf",
                big32[i]
            );
        }
    }
}

// The guard sits at `int_cap` (172 / 36) while the true overflow points are 171.624 and
// 35.040. The gap is covered by the arithmetic, not by the guard, so it needs its own
// check, since a guard placed one integer too low would silently clip finite answers here.
#[test]
fn tgamma_is_finite_right_up_to_the_overflow_point() {
    let last64 = [171.0f64, 171.5, 171.6, 171.62];
    for (name, got) in [
        ("DefaultPolicy", f64x4::new(last64).tgamma().into_array()),
        ("Precision", f64x4::new(last64).tgamma_p::<Precision>().into_array()),
    ] {
        for (i, v) in got.iter().enumerate() {
            assert!(v.is_finite(), "f64 {name}: tgamma({}) = {v}, want finite", last64[i]);
            assert!(
                close(*v, libm::tgamma(last64[i]), 1.0e-6),
                "f64 {name}: tgamma({})",
                last64[i]
            );
        }
    }

    let last32 = [34.0f32, 34.5, 34.9, 35.0, 35.02, 33.0, 32.0, 30.0];
    for (name, got) in [
        ("DefaultPolicy", f32x8::new(last32).tgamma().into_array()),
        ("Precision", f32x8::new(last32).tgamma_p::<Precision>().into_array()),
    ] {
        for (i, v) in got.iter().enumerate() {
            assert!(v.is_finite(), "f32 {name}: tgamma({}) = {v}, want finite", last32[i]);
            assert!(
                close(*v as f64, libm::tgammaf(last32[i]) as f64, 3.0e-3),
                "f32 {name}: tgamma({}) = {v}",
                last32[i]
            );
        }
    }
}

// Reflected lanes: `Gamma` of a large negative non-integer is far below the smallest
// subnormal, so saturating to zero is the right answer. The overflow guard drives the
// reflected divisor to infinity, which lands there. Checked because it is a
// consequence of the guard rather than something written directly.
#[test]
fn tgamma_of_large_negative_saturates_to_zero() {
    let neg = [-180.5f64, -200.25, -1.0e10 - 0.5, -400.75];
    for (name, got) in [
        ("DefaultPolicy", f64x4::new(neg).tgamma().into_array()),
        ("Precision", f64x4::new(neg).tgamma_p::<Precision>().into_array()),
    ] {
        for (i, v) in got.iter().enumerate() {
            assert!(*v == 0.0, "f64 {name}: tgamma({}) = {v}, want +-0", neg[i]);
        }
    }
}

#[test]
fn lgamma_f32() {
    sweep_f32!("lgamma", lgamma, libm::lgammaf, 0.1, 40.0, 3.0e-3);
}
#[test]
fn lgamma_f64() {
    sweep_f64!("lgamma", lgamma, libm::lgamma, 0.1, 40.0, 1.0e-6);
}

// Exercise the `Best`-precision branches of the f64 gamma port (reflection for
// negatives, the integer fast-path, the (0,1) recurrence shift, and tiny-value
// handling) which the default `Performance` sweeps above do not reach.
#[test]
fn tgamma_f64_precision_wide() {
    let n = 4000usize;
    let mut buf = [0.0f64; 4];
    let mut i = 0;
    while i < n {
        for k in 0..4 {
            let t = ((i + k) as f64) / (n as f64);
            buf[k] = -8.0 + t * 16.0; // [-8, 8], includes negatives and integers
        }
        let got = f64x4::new(buf).tgamma_p::<Precision>().into_array();
        for k in 0..4 {
            let want = libm::tgamma(buf[k]);
            assert!(
                close(got[k], want, 1.0e-9),
                "tgamma f64 (Precision) @ x={}: got {}, libm {}",
                buf[k],
                got[k],
                want
            );
        }
        i += 4;
    }
}

#[test]
fn lgamma_f64_precision_wide() {
    let n = 4000usize;
    let mut buf = [0.0f64; 4];
    let mut i = 0;
    while i < n {
        for k in 0..4 {
            let t = ((i + k) as f64) / (n as f64);
            buf[k] = -20.0 + t * 60.0; // [-20, 40], includes negatives
        }
        let got = f64x4::new(buf).lgamma_p::<Precision>().into_array();
        for k in 0..4 {
            let want = libm::lgamma(buf[k]);
            // skip near the negative-integer poles where lgamma -> +inf
            if !want.is_finite() {
                continue;
            }
            assert!(
                close(got[k], want, 1.0e-9),
                "lgamma f64 (Precision) @ x={}: got {}, libm {}",
                buf[k],
                got[k],
                want
            );
        }
        i += 4;
    }
}

#[test]
fn tgamma_f32_precision_wide() {
    let n = 4000usize;
    let mut buf = [0.0f32; 8];
    let mut i = 0;
    while i < n {
        for k in 0..8 {
            let t = ((i + k) as f32) / (n as f32);
            buf[k] = -8.0 + t * 16.0; // [-8, 8], includes negatives and integers
        }
        let got = f32x8::new(buf).tgamma_p::<Precision>().into_array();
        for k in 0..8 {
            let want = libm::tgammaf(buf[k]);
            assert!(
                close(got[k] as f64, want as f64, 5.0e-5),
                "tgamma f32 (Precision) @ x={}: got {}, libm {}",
                buf[k],
                got[k],
                want
            );
        }
        i += 8;
    }
}

// The default `Performance` policy takes the low-precision Pade branch in the f32
// `lgamma_r`, so the f32 Lanczos path is only reachable at `Best` precision.
#[test]
fn lgamma_f32_precision_wide() {
    let n = 4000usize;
    let mut buf = [0.0f32; 8];
    let mut i = 0;
    while i < n {
        for k in 0..8 {
            let t = ((i + k) as f32) / (n as f32);
            buf[k] = -20.0 + t * 60.0; // [-20, 40], includes negatives
        }
        let got = f32x8::new(buf).lgamma_p::<Precision>().into_array();
        for k in 0..8 {
            let want = libm::lgammaf(buf[k]);
            if !want.is_finite() {
                continue;
            }
            assert!(
                close(got[k] as f64, want as f64, 3.0e-4),
                "lgamma f32 (Precision) @ x={}: got {}, libm {}",
                buf[k],
                got[k],
                want
            );
        }
        i += 8;
    }
}

// B(a, b) = exp(lgamma(a) + lgamma(b) - lgamma(a + b)) for a, b > 0.
fn beta_ref(a: f64, b: f64) -> f64 {
    (libm::lgamma(a) + libm::lgamma(b) - libm::lgamma(a + b)).exp()
}

#[test]
fn beta_f32() {
    let mut a = [0.0f32; 8];
    let mut b = [0.0f32; 8];
    for i in 0..64 {
        for k in 0..8 {
            a[k] = 0.25 + (i as f32) * 0.125;
            b[k] = 0.25 + (k as f32) * 0.75;
        }
        let got = f32x8::new(a).beta(f32x8::new(b)).into_array();
        for k in 0..8 {
            let want = beta_ref(a[k] as f64, b[k] as f64);
            assert!(
                close(got[k] as f64, want, 1.0e-5),
                "beta f32 @ a={}, b={}: got {}, ref {}",
                a[k],
                b[k],
                got[k],
                want
            );
        }
    }
}

#[test]
fn beta_f64() {
    let mut a = [0.0f64; 4];
    let mut b = [0.0f64; 4];
    for i in 0..64 {
        for k in 0..4 {
            a[k] = 0.25 + (i as f64) * 0.125;
            b[k] = 0.25 + (k as f64) * 0.75;
        }
        let got = f64x4::new(a).beta(f64x4::new(b)).into_array();
        for k in 0..4 {
            let want = beta_ref(a[k], b[k]);
            assert!(
                close(got[k], want, 1.0e-12),
                "beta f64 @ a={}, b={}: got {}, ref {}",
                a[k],
                b[k],
                got[k],
                want
            );
        }
    }
}

// libm has no digamma, so use a high-accuracy scalar reference: shift the argument
// up to x >= 12 with the recurrence psi(x) = psi(x+1) - 1/x (valid for any non-pole x,
// including negative non-integers), then the Bernoulli asymptotic series. Independent
// of the implementation under test (different threshold, more series terms).
fn digamma_ref(mut x: f64) -> f64 {
    let mut result = 0.0;
    while x < 12.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result += x.ln() - 0.5 * inv;
    let mut t = inv2;
    result -= t / 12.0; // B2/(2)  = 1/12
    t *= inv2;
    result += t / 120.0; // -B4/4  = 1/120
    t *= inv2;
    result -= t / 252.0; // B6/6   = 1/252
    t *= inv2;
    result += t / 240.0; // -B8/8  = 1/240
    t *= inv2;
    result -= t / 132.0; // B10/10 = 1/132
    result
}

#[test]
fn digamma_f64() {
    let n = 4000usize;
    let mut buf = [0.0f64; 4];
    let mut i = 0;
    while i < n {
        for k in 0..4 {
            let t = ((i + k) as f64) / (n as f64);
            buf[k] = 0.05 + t * (20.0 - 0.05); // positive domain (pole only at 0)
        }
        let got = f64x4::new(buf).digamma().into_array();
        for k in 0..4 {
            let want = digamma_ref(buf[k]);
            assert!(
                close(got[k], want, 1.0e-6),
                "digamma f64 @ x={}: got {}, ref {}",
                buf[k],
                got[k],
                want
            );
        }
        i += 4;
    }
}

// Exercises the reflection path (x <= -1) at Precision, skipping the integer poles.
#[test]
fn digamma_f64_reflection() {
    let n = 4000usize;
    let mut buf = [0.0f64; 4];
    let mut i = 0;
    while i < n {
        for k in 0..4 {
            let t = ((i + k) as f64) / (n as f64);
            buf[k] = -8.0 + t * 7.0; // [-8, -1]
        }
        let got = f64x4::new(buf).digamma_p::<Precision>().into_array();
        for k in 0..4 {
            // skip within 0.02 of a negative integer (digamma pole)
            if (buf[k] - buf[k].round()).abs() < 0.02 {
                continue;
            }
            let want = digamma_ref(buf[k]);
            assert!(
                close(got[k], want, 1.0e-7),
                "digamma f64 reflect @ x={}: got {}, ref {}",
                buf[k],
                got[k],
                want
            );
        }
        i += 4;
    }
}

// `trigamma` is public on `SpecialMath` as of the polygamma arc. These tests use the
// public `_p` spelling (importing `SpecializedSpecialMath` at file scope would make
// every method the two traits share ambiguous).
mod trigamma {
    use super::*;

    // psi_1 reference: recurrence psi_1(x) = 1/x^2 + psi_1(x+1) walks any x (including
    // negative non-integers) up past the poles, then the Bernoulli asymptotic series
    //   psi_1(x) ~ 1/x + 1/(2x^2) + sum_{k>=1} B_2k / x^(2k+1).
    // Independent of the implementation under test: no minimax rationals, and it never
    // touches the reflection identity, so the reflection test below is a real check.
    fn trigamma_ref(mut x: f64) -> f64 {
        let mut result = 0.0;
        while x < 20.0 {
            result += 1.0 / (x * x);
            x += 1.0;
        }
        let inv = 1.0 / x;
        let inv2 = inv * inv;
        result += inv + 0.5 * inv2;
        let mut t = inv2 * inv; // 1/x^3
        result += t / 6.0; // B2  =  1/6
        t *= inv2;
        result -= t / 30.0; // B4  = -1/30
        t *= inv2;
        result += t / 42.0; // B6  =  1/42
        t *= inv2;
        result -= t / 30.0; // B8  = -1/30
        t *= inv2;
        result += t * 5.0 / 66.0; // B10 =  5/66
        result
    }

    // Closed forms: psi_1(1) = pi^2/6, psi_1(2) = pi^2/6 - 1, psi_1(1/2) = pi^2/2.
    #[test]
    fn trigamma_f64_exact_points() {
        let pi2 = core::f64::consts::PI * core::f64::consts::PI;
        let buf = [1.0, 2.0, 0.5, 3.0];
        let want = [pi2 / 6.0, pi2 / 6.0 - 1.0, pi2 / 2.0, pi2 / 6.0 - 1.25];
        let got = f64x4::new(buf).trigamma_p::<Precision>().into_array();
        for k in 0..4 {
            assert!(
                close(got[k], want[k], 1.0e-14),
                "trigamma f64 @ x={}: got {}, exact {}",
                buf[k],
                got[k],
                want[k]
            );
        }
    }

    #[test]
    fn trigamma_f64() {
        let n = 4000usize;
        let mut buf = [0.0f64; 4];
        let mut i = 0;
        while i < n {
            for k in 0..4 {
                let t = ((i + k) as f64) / (n as f64);
                buf[k] = 0.05 + t * (20.0 - 0.05); // positive domain (pole only at 0)
            }
            let got = f64x4::new(buf).trigamma_p::<DefaultPolicy>().into_array();
            for k in 0..4 {
                let want = trigamma_ref(buf[k]);
                assert!(
                    close(got[k], want, 1.0e-13),
                    "trigamma f64 @ x={}: got {}, ref {}",
                    buf[k],
                    got[k],
                    want
                );
            }
            i += 4;
        }
    }

    // Exercises the reflection path (x <= 0), skipping the double poles at the negative
    // integers, plus the single-step recurrence on (0, 1).
    #[test]
    fn trigamma_f64_reflection() {
        let n = 4000usize;
        let mut buf = [0.0f64; 4];
        let mut i = 0;
        while i < n {
            for k in 0..4 {
                let t = ((i + k) as f64) / (n as f64);
                buf[k] = -8.0 + t * 8.0; // [-8, 0]
            }
            let got = f64x4::new(buf).trigamma_p::<Precision>().into_array();
            for k in 0..4 {
                if (buf[k] - buf[k].round()).abs() < 0.02 {
                    continue;
                }
                let want = trigamma_ref(buf[k]);
                assert!(
                    close(got[k], want, 1.0e-12),
                    "trigamma f64 reflect @ x={}: got {}, ref {}",
                    buf[k],
                    got[k],
                    want
                );
            }
            i += 4;
        }
    }

    // The poles: psi_1 has a double pole at 0 and the negative integers, so unlike psi
    // both one-sided limits are +inf.
    #[test]
    fn trigamma_f64_poles() {
        let buf = [0.0, -1.0, -2.0, -7.0];
        let got = f64x4::new(buf).trigamma_p::<Precision>().into_array();
        for k in 0..4 {
            assert!(
                got[k] == f64::INFINITY,
                "trigamma f64 pole @ x={}: got {}, want +inf",
                buf[k],
                got[k]
            );
        }
    }

    #[test]
    fn trigamma_f32() {
        let n = 4000usize;
        let mut buf = [0.0f32; 8];
        let mut i = 0;
        while i < n {
            for k in 0..8 {
                let t = ((i + k) as f32) / (n as f32);
                buf[k] = 0.05 + t * (20.0 - 0.05);
            }
            let got = f32x8::new(buf).trigamma_p::<DefaultPolicy>().into_array();
            for k in 0..8 {
                let want = trigamma_ref(buf[k] as f64);
                assert!(
                    close(got[k] as f64, want, 1.0e-6),
                    "trigamma f32 @ x={}: got {}, ref {}",
                    buf[k],
                    got[k],
                    want
                );
            }
            i += 8;
        }
    }
}

#[test]
fn digamma_f32() {
    let n = 4000usize;
    let mut buf = [0.0f32; 8];
    let mut i = 0;
    while i < n {
        for k in 0..8 {
            let t = ((i + k) as f32) / (n as f32);
            buf[k] = 0.05 + t * (20.0 - 0.05);
        }
        let got = f32x8::new(buf).digamma().into_array();
        for k in 0..8 {
            let want = digamma_ref(buf[k] as f64);
            assert!(
                close(got[k] as f64, want, 2.0e-3),
                "digamma f32 @ x={}: got {}, ref {}",
                buf[k],
                got[k],
                want
            );
        }
        i += 8;
    }
}

// Round-trip check: the standard-normal CDF of probit(p) must recover p.
// cdf(x) = 0.5 * (1 + erf(x / sqrt(2))), using libm::erf as the oracle.
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x * core::f64::consts::FRAC_1_SQRT_2))
}

#[test]
fn probit_f64() {
    let n = 4000usize;
    let mut buf = [0.0f64; 4];
    let mut i = 0;
    while i < n {
        for k in 0..4 {
            let t = ((i + k) as f64) / (n as f64);
            buf[k] = 1.0e-6 + t * (1.0 - 2.0e-6); // p in (0, 1), away from the tails
        }
        let got = f64x4::new(buf).probit_p::<Precision>().into_array();
        for k in 0..4 {
            let round_trip = normal_cdf(got[k]);
            assert!(
                close(round_trip, buf[k], 1.0e-9),
                "probit f64 @ p={}: probit={}, cdf(probit)={}",
                buf[k],
                got[k],
                round_trip
            );
        }
        i += 4;
    }
}

// Round-trip check: erf(erfinv(y)) must recover y, using libm::erf as the oracle.
// Sweeps into the tails (|y| -> 1) where the previous fit degraded badly.
#[test]
fn erfinv_f64() {
    let n = 4000usize;
    let mut buf = [0.0f64; 4];
    let mut i = 0;
    while i < n {
        for k in 0..4 {
            let t = ((i + k) as f64) / (n as f64);
            buf[k] = -0.999999 + t * (2.0 * 0.999999); // (-1, 1), into the tails
        }
        let got = f64x4::new(buf).erfinv_p::<Precision>().into_array();
        for k in 0..4 {
            let round_trip = libm::erf(got[k]);
            assert!(
                close(round_trip, buf[k], 1.0e-9),
                "erfinv f64 @ y={}: erfinv={}, erf(erfinv)={}",
                buf[k],
                got[k],
                round_trip
            );
        }
        i += 4;
    }
}

// Default (Performance) policy uses a single Halley step, so verify it is still solid.
#[test]
fn erfinv_f64_default() {
    let n = 4000usize;
    let mut buf = [0.0f64; 4];
    let mut i = 0;
    while i < n {
        for k in 0..4 {
            let t = ((i + k) as f64) / (n as f64);
            buf[k] = -0.9999 + t * (2.0 * 0.9999);
        }
        let got = f64x4::new(buf).erfinv().into_array();
        for k in 0..4 {
            let round_trip = libm::erf(got[k]);
            assert!(
                close(round_trip, buf[k], 1.0e-6),
                "erfinv f64 (default) @ y={}: erfinv={}, erf(erfinv)={}",
                buf[k],
                got[k],
                round_trip
            );
        }
        i += 4;
    }
}

// Exercises the kind-dispatched public entry point SpecialMath::ellint with the request structs,
// end-to-end through the trait plumbing (vector path).
#[test]
fn ellint_public_api() {
    use thermite_special::elliptic::{EllintE, EllintEInc, EllintF, EllintK};

    // Complete: K(0.5), E(0.5).
    let k = f64x4::splat(0.5);
    let kk = f64x4::ellint_p::<Precision, _>(EllintK { k }).into_array();
    let ee = f64x4::ellint_p::<Precision, _>(EllintE { k }).into_array();
    assert!(close(kk[0], 1.685750354812596, 1.0e-12), "K(0.5) = {}", kk[0]);
    assert!(close(ee[0], 1.467462209339427, 1.0e-12), "E(0.5) = {}", ee[0]);

    // Incomplete: F(phi, k) and E(phi, k) from the reference tables.
    let phi = f64x4::splat(1.302990057703935);
    let k2 = f64x4::splat(0.1279518954120547);
    let f = f64x4::ellint_p::<Precision, _>(EllintF { phi, k: k2 }).into_array();
    let e = f64x4::ellint_p::<Precision, _>(EllintEInc { phi, k: k2 }).into_array();
    assert!(close(f[0], 1.307312511398114, 1.0e-12), "F(phi,k) = {}", f[0]);
    assert!(close(e[0], 1.298690225567921, 1.0e-12), "E(phi,k) = {}", e[0]);
}

// Exercises the public Carlson entry point (SpecialMath::carlson + CarlsonKind request structs)
// through the crate's re-export, confirming the type-dispatched API and per-kind arity are reachable.
#[test]
fn carlson_public_api() {
    use thermite_special::elliptic::{CarlsonRc, CarlsonRf, CarlsonRg, CarlsonRj};

    let q = f64x4::splat(4.0);
    // R_F(4,4,4)=1/2, R_G(4,4,4)=2, R_C(4,4)=1/2, R_J(4,4,4,4)=1/8 (each its own named fields).
    let rf = f64x4::carlson_p::<Precision, _>(CarlsonRf { x: q, y: q, z: q }).into_array();
    let rg = f64x4::carlson_p::<Precision, _>(CarlsonRg { x: q, y: q, z: q }).into_array();
    let rc = f64x4::carlson_p::<Precision, _>(CarlsonRc { x: q, y: q }).into_array();
    let rj = f64x4::carlson_p::<Precision, _>(CarlsonRj { x: q, y: q, z: q, p: q }).into_array();
    assert!(close(rf[0], 0.5, 1.0e-13), "R_F = {}", rf[0]);
    assert!(close(rg[0], 2.0, 1.0e-13), "R_G = {}", rg[0]);
    assert!(close(rc[0], 0.5, 1.0e-13), "R_C = {}", rc[0]);
    assert!(close(rj[0], 0.125, 1.0e-13), "R_J = {}", rj[0]);
    // Default-policy entry, V inferred from the request struct.
    let rf_def = f64x4::carlson(CarlsonRf { x: q, y: q, z: q }).into_array();
    assert!(close(rf_def[0], 0.5, 1.0e-13), "R_F default = {}", rf_def[0]);

    // Scalar path through the Scalar* trait (wrap -> vector eval -> unwrap).
    use thermite_special::ScalarSpecialMath as _;
    let rf_s = f64::scalar_carlson(CarlsonRf {
        x: 4.0_f64,
        y: 4.0,
        z: 4.0,
    });
    assert!(close(rf_s, 0.5, 1.0e-13), "scalar R_F = {rf_s}");
}

#[test]
fn logistic_sigmoid_f32() {
    sweep_f32!(
        "sigmoid",
        logistic_sigmoid,
        |x: f32| 1.0 / (1.0 + (-x).exp()),
        -30.0,
        30.0,
        2.0e-3
    );
}
#[test]
fn logistic_sigmoid_f64() {
    sweep_f64!(
        "sigmoid",
        logistic_sigmoid,
        |x: f64| 1.0 / (1.0 + (-x).exp()),
        -30.0,
        30.0,
        1.0e-6
    );
}
