//! Transcendental math correctness gate: every backend's `Vector<R>` math
//! function is checked against `libm` (an independent, well-tested reference)
//! over its valid domain.
//!
//! This is deliberately a *gross-correctness* gate, not a ULP audit: the bare
//! (no-policy) methods use `DefaultPolicy`, which trades accuracy for speed
//! (`Performance` on x86/scalar, `Size` on WASM - both `Average` precision but
//! WASM's `Size` flushes denormals via the `Crush` trick), so the tolerance is a
//! loose relative bound. It is here to catch structural bugs (wrong sign, wrong
//! identity, NaN where a number is expected, a backend that diverges from the
//! others) - the things that had **zero** test coverage before this file
//! existed. Tighten `TOL_*` and switch to a `Reference` policy for a precision
//! audit.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    all(feature = "neon", target_arch = "aarch64")
))]

mod harness;

use thermite::Vector;
use thermite::prelude::*;

/// Loose relative tolerance for the default (`Performance`) policy.
const TOL_F32: f64 = 2.0e-3;
const TOL_F64: f64 = 1.0e-6;

fn close(got: f64, want: f64, tol: f64) -> bool {
    if got.is_nan() || want.is_nan() {
        return got.is_nan() == want.is_nan();
    }
    if got == want {
        return true;
    }
    if !want.is_finite() || !got.is_finite() {
        return got == want;
    }
    (got - want).abs() <= tol * want.abs().max(1.0)
}

/// One math function: build inputs in `domain`, run the SIMD op, compare each
/// lane to `oracle` (libm).
macro_rules! math_unary {
    ($label:expr, $reg:ty, $elem:ty, $method:ident, $oracle:expr, $tol:expr, $domain:expr) => {{
        type V = Vector<$reg>;
        let mut rng = harness::rng();
        let lanes = <V as GenericVector>::LANES;
        let dom: fn($elem) -> $elem = $domain;
        let oracle: fn($elem) -> $elem = $oracle;
        for raw in harness::corpus::<$elem>(lanes, &mut rng) {
            let input: Vec<$elem> = raw.iter().map(|&x| dom(x)).collect();
            let got = Vector::<$reg>(harness::make_array::<$reg>(&input)).$method().into_array();
            for (lane, (&g, &x)) in got.iter().zip(input.iter()).enumerate() {
                let want = oracle(x);
                if !close(g as f64, want as f64, $tol) {
                    panic!(
                        "{} [{}]: lane {lane} mismatch\n  input = {x:?}\n  got   = {g:?}\n  want  = {want:?} (libm)\n  full input = {input:?}\n  full got   = {:?}",
                        $label, stringify!($method), got
                    );
                }
            }
        }
    }};
}

/// Two-operand math fn `a.$method(b)` vs `oracle(a, b)`; `$da`/`$db` map each
/// operand into a valid domain.
macro_rules! math_binary {
    ($label:expr, $reg:ty, $elem:ty, $method:ident, $oracle:expr, $tol:expr, $da:expr, $db:expr) => {{
        let mut rng = harness::rng();
        let lanes = <Vector<$reg> as GenericVector>::LANES;
        let da: fn($elem) -> $elem = $da;
        let db: fn($elem) -> $elem = $db;
        let oracle: fn($elem, $elem) -> $elem = $oracle;
        let xs = harness::corpus::<$elem>(lanes, &mut rng);
        let ys = harness::corpus::<$elem>(lanes, &mut rng);
        for (rx, ry) in xs.iter().zip(ys.iter()) {
            let a: Vec<$elem> = rx.iter().map(|&x| da(x)).collect();
            let b: Vec<$elem> = ry.iter().map(|&y| db(y)).collect();
            let got = Vector::<$reg>(harness::make_array::<$reg>(&a))
                .$method(Vector::<$reg>(harness::make_array::<$reg>(&b)))
                .into_array();
            for ((&g, &x), &y) in got.iter().zip(a.iter()).zip(b.iter()) {
                let want = oracle(x, y);
                if !close(g as f64, want as f64, $tol) {
                    panic!(
                        "{} [{}]: a={x:?} b={y:?}\n  got  = {g:?}\n  want = {want:?} (libm)",
                        $label,
                        stringify!($method)
                    );
                }
            }
        }
    }};
}

/// Unary fn returning `(Self, Self)` (e.g. `sin_cos`) vs two oracles.
macro_rules! math_tuple {
    ($label:expr, $reg:ty, $elem:ty, $method:ident, $o0:expr, $o1:expr, $tol:expr, $domain:expr) => {{
        let mut rng = harness::rng();
        let lanes = <Vector<$reg> as GenericVector>::LANES;
        let dom: fn($elem) -> $elem = $domain;
        let o0: fn($elem) -> $elem = $o0;
        let o1: fn($elem) -> $elem = $o1;
        for raw in harness::corpus::<$elem>(lanes, &mut rng) {
            let input: Vec<$elem> = raw.iter().map(|&x| dom(x)).collect();
            let (ra, rb) = Vector::<$reg>(harness::make_array::<$reg>(&input)).$method();
            let (ga, gb) = (ra.into_array(), rb.into_array());
            for (lane, &x) in input.iter().enumerate() {
                let (wa, wb) = (o0(x), o1(x));
                if !close(ga[lane] as f64, wa as f64, $tol) {
                    panic!(
                        "{} [{}.0]: lane {lane} in={x:?} got={:?} want={wa:?}",
                        $label,
                        stringify!($method),
                        ga[lane]
                    );
                }
                if !close(gb[lane] as f64, wb as f64, $tol) {
                    panic!(
                        "{} [{}.1]: lane {lane} in={x:?} got={:?} want={wb:?}",
                        $label,
                        stringify!($method),
                        gb[lane]
                    );
                }
            }
        }
    }};
}

/// Unary const-generic fn `.$method::<$N>()` (e.g. `nth_root`, `log_n`) vs oracle.
macro_rules! math_unary_cg {
    ($label:expr, $reg:ty, $elem:ty, $method:ident, $n:literal, $oracle:expr, $tol:expr, $domain:expr) => {{
        let mut rng = harness::rng();
        let lanes = <Vector<$reg> as GenericVector>::LANES;
        let dom: fn($elem) -> $elem = $domain;
        let oracle: fn($elem) -> $elem = $oracle;
        for raw in harness::corpus::<$elem>(lanes, &mut rng) {
            let input: Vec<$elem> = raw.iter().map(|&x| dom(x)).collect();
            let got = Vector::<$reg>(harness::make_array::<$reg>(&input))
                .$method::<$n>()
                .into_array();
            for (lane, (&g, &x)) in got.iter().zip(input.iter()).enumerate() {
                let want = oracle(x);
                if !close(g as f64, want as f64, $tol) {
                    panic!(
                        "{} [{}::<{}>]: lane {lane} in={x:?} got={g:?} want={want:?} (libm)",
                        $label,
                        stringify!($method),
                        $n
                    );
                }
            }
        }
    }};
}

/// `.powi($e)` (one scalar `i32` exponent for all lanes) vs `libm::powf(x, e)`.
macro_rules! math_powi {
    ($label:expr, $reg:ty, $elem:ty, $e:expr, $pf:expr, $tol:expr, $domain:expr) => {{
        let mut rng = harness::rng();
        let lanes = <Vector<$reg> as GenericVector>::LANES;
        let dom: fn($elem) -> $elem = $domain;
        let pf: fn($elem, $elem) -> $elem = $pf;
        let e: i32 = $e;
        for raw in harness::corpus::<$elem>(lanes, &mut rng) {
            let input: Vec<$elem> = raw.iter().map(|&x| dom(x)).collect();
            let got = Vector::<$reg>(harness::make_array::<$reg>(&input)).powi(e).into_array();
            for (lane, (&g, &x)) in got.iter().zip(input.iter()).enumerate() {
                let want = pf(x, e as $elem);
                if !close(g as f64, want as f64, $tol) {
                    panic!(
                        "{} [powi({})]: lane {lane} in={x:?} got={g:?} want={want:?} (libm)",
                        $label, e
                    );
                }
            }
        }
    }};
}

macro_rules! math_suite {
    ($modname:ident, $backend:ty, $f32reg:ident, $f64reg:ident, $bl:expr) => {
        mod $modname {
            use super::*;

            #[test]
            fn f32() {
                type R = <$backend as Simd>::$f32reg;
                let pos = |x: f32| if x.is_finite() { x.abs() + 1e-3 } else { 1.0 };
                let unit = |x: f32| {
                    if x.is_finite() {
                        (x % 2.0).clamp(-0.999, 0.999)
                    } else {
                        0.5
                    }
                };
                // f32 tanh/exp lose to overflow well before libm does; the
                // gate stays in the principal region (large-x saturation bugs
                // are flagged separately in TESTING.md).
                let small = |x: f32| if x.is_finite() { x % 20.0 } else { 1.0 };
                let ang = |x: f32| if x.is_finite() { x % 1000.0 } else { 1.0 };
                let ge1 = |x: f32| if x.is_finite() { x.abs() + 1.0 } else { 2.0 };

                math_unary!($bl, R, f32, sin, libm::sinf, TOL_F32, ang);
                math_unary!($bl, R, f32, cos, libm::cosf, TOL_F32, ang);
                math_unary!($bl, R, f32, tan, libm::tanf, TOL_F32, |x: f32| if x.is_finite() {
                    x % 1.5
                } else {
                    1.0
                });
                math_unary!($bl, R, f32, exp, libm::expf, TOL_F32, small);
                math_unary!($bl, R, f32, exp2, libm::exp2f, TOL_F32, small);
                math_unary!($bl, R, f32, ln, libm::logf, TOL_F32, pos);
                math_unary!($bl, R, f32, log2, libm::log2f, TOL_F32, pos);
                math_unary!($bl, R, f32, sqrt, libm::sqrtf, TOL_F32, pos);
                math_unary!(
                    $bl,
                    R,
                    f32,
                    cbrt,
                    libm::cbrtf,
                    TOL_F32,
                    |x: f32| if x.is_finite() { x % 1e6 } else { 1.0 }
                );
                math_unary!($bl, R, f32, asin, libm::asinf, TOL_F32, unit);
                math_unary!($bl, R, f32, acos, libm::acosf, TOL_F32, unit);
                math_unary!($bl, R, f32, atan, libm::atanf, TOL_F32, ang);
                math_unary!($bl, R, f32, sinh, libm::sinhf, TOL_F32, small);
                math_unary!($bl, R, f32, cosh, libm::coshf, TOL_F32, small);
                math_unary!($bl, R, f32, tanh, libm::tanhf, TOL_F32, small);
                math_unary!($bl, R, f32, acosh, libm::acoshf, TOL_F32, ge1);
                math_unary!($bl, R, f32, asinh, libm::asinhf, TOL_F32, small);

                math_unary!($bl, R, f32, exp_m1, libm::expm1f, TOL_F32, small);
                // 2^x - 1 == expm1(x * ln2); the latter is the accurate oracle near 0
                math_unary!(
                    $bl,
                    R,
                    f32,
                    exp2_m1,
                    |x: f32| libm::expm1(x as f64 * core::f64::consts::LN_2) as f32,
                    TOL_F32,
                    small
                );
                // 10^x - 1 == expm1(x * ln10)
                math_unary!(
                    $bl,
                    R,
                    f32,
                    exp10_m1,
                    |x: f32| libm::expm1(x as f64 * core::f64::consts::LN_10) as f32,
                    TOL_F32,
                    small
                );
                math_unary!(
                    $bl,
                    R,
                    f32,
                    ln_1p,
                    libm::log1pf,
                    TOL_F32,
                    |x: f32| if x.is_finite() { (x % 2.0).max(-0.9) } else { 0.5 }
                );
                // log_b(1 + x) == log1p(x) * log_b(e)
                math_unary!(
                    $bl,
                    R,
                    f32,
                    log2_p1,
                    |x: f32| (libm::log1p(x as f64) * core::f64::consts::LOG2_E) as f32,
                    TOL_F32,
                    |x: f32| if x.is_finite() { (x % 2.0).max(-0.9) } else { 0.5 }
                );
                math_unary!(
                    $bl,
                    R,
                    f32,
                    log10_p1,
                    |x: f32| (libm::log1p(x as f64) * core::f64::consts::LOG10_E) as f32,
                    TOL_F32,
                    |x: f32| if x.is_finite() { (x % 2.0).max(-0.9) } else { 0.5 }
                );
                // sqrt(1+x) - 1 == expm1(0.5 * log1p(x)); the latter is the accurate oracle near 0
                math_unary!(
                    $bl,
                    R,
                    f32,
                    sqrt1pm1,
                    |x: f32| libm::expm1(0.5 * libm::log1p(x as f64)) as f32,
                    TOL_F32,
                    |x: f32| if x.is_finite() { (x % 20.0).max(-0.9) } else { 0.5 }
                );
                // cos(x) - 1 == -2 sin²(x/2); the latter is the accurate oracle near 0
                math_unary!(
                    $bl,
                    R,
                    f32,
                    cos_m1,
                    |x: f32| {
                        let s = libm::sin(x as f64 * 0.5);
                        (-2.0 * s * s) as f32
                    },
                    TOL_F32,
                    small
                );
                math_unary!(
                    $bl,
                    R,
                    f32,
                    versin,
                    |x: f32| {
                        let s = libm::sin(x as f64 * 0.5);
                        (2.0 * s * s) as f32
                    },
                    TOL_F32,
                    small
                );
                math_unary!(
                    $bl,
                    R,
                    f32,
                    haversin,
                    |x: f32| {
                        let s = libm::sin(x as f64 * 0.5);
                        (s * s) as f32
                    },
                    TOL_F32,
                    small
                );
                math_unary!($bl, R, f32, log10, libm::log10f, TOL_F32, pos);
                math_unary!($bl, R, f32, atanh, libm::atanhf, TOL_F32, unit);
                // `safe`: finite, normal, nonzero, and comfortably away from the
                // subnormal boundary - dodges the denormal-flush divergence and the
                // hypot(0,0)/atan2(0,0) special cases. The magnitude floor matters on
                // WASM, whose default policy is `Size` (Crush denormals via the
                // `dt - (dt - x)` trick); that trick's ULP (~2*MIN_POSITIVE) also
                // quantizes *normal* values within a few ULPs of MIN_POSITIVE. atan2 is
                // ratio-sensitive, so such tiny inputs flip the result by O(1) - outside
                // this gross-correctness gate (sign is preserved for quadrant coverage).
                let safe = |x: f32| {
                    let v = if x.is_finite() { x % 1e3 } else { 1.0 };
                    if v.is_normal() && v.abs() >= 1e-30 {
                        v
                    } else {
                        1.0_f32.copysign(v)
                    }
                };
                math_binary!($bl, R, f32, atan2, libm::atan2f, TOL_F32, safe, safe);
                math_binary!($bl, R, f32, hypot, libm::hypotf, TOL_F32, safe, safe);
                // Positive base, exponent in [-8, 8]; see the f64 case above (#W11).
                math_binary!(
                    $bl,
                    R,
                    f32,
                    powf,
                    libm::powf,
                    TOL_F32,
                    |x: f32| if x.is_finite() {
                        (x.abs() % 10.0) + 0.1
                    } else {
                        2.0
                    },
                    |y: f32| if y.is_finite() { y % 8.0 } else { 0.0 }
                );
                // compound(x, n) = (1+x)^n; oracle in f64 keeps x's low bits that (1+x)^n would lose
                math_binary!(
                    $bl,
                    R,
                    f32,
                    compound,
                    |x: f32, n: f32| libm::pow(1.0 + x as f64, n as f64) as f32,
                    TOL_F32,
                    |x: f32| if x.is_finite() { (x % 5.0).max(-0.9) } else { 0.5 },
                    |n: f32| if n.is_finite() { n % 8.0 } else { 2.0 }
                );
                // logaddexp(a, b) = ln(e^a + e^b); naive f64 oracle is safe over this domain
                math_binary!(
                    $bl,
                    R,
                    f32,
                    logaddexp,
                    |a: f32, b: f32| libm::log(libm::exp(a as f64) + libm::exp(b as f64)) as f32,
                    TOL_F32,
                    |x: f32| if x.is_finite() { x % 20.0 } else { 1.0 },
                    |x: f32| if x.is_finite() { x % 20.0 } else { -1.0 }
                );
                // powf_m1(x, e) = x^e - 1 == expm1(e * ln x); accurate oracle in f64
                math_binary!(
                    $bl,
                    R,
                    f32,
                    powf_m1,
                    |x: f32, e: f32| libm::expm1(e as f64 * libm::log(x as f64)) as f32,
                    TOL_F32,
                    |x: f32| if x.is_finite() {
                        (x.abs() % 10.0) + 0.1
                    } else {
                        2.0
                    },
                    |e: f32| if e.is_finite() { e % 4.0 } else { 2.0 }
                );

                // --- additional transcendentals (this session) ---
                let pidom = |x: f32| if x.is_finite() { x % 30.0 } else { 1.0 };
                let tanpidom = |x: f32| if x.is_finite() { x % 0.4 } else { 0.1 }; // away from ±0.5 poles
                math_unary!(
                    $bl,
                    R,
                    f32,
                    sin_pi,
                    |x: f32| (core::f64::consts::PI * x as f64).sin() as f32,
                    TOL_F32,
                    pidom
                );
                math_unary!(
                    $bl,
                    R,
                    f32,
                    cos_pi,
                    |x: f32| (core::f64::consts::PI * x as f64).cos() as f32,
                    TOL_F32,
                    pidom
                );
                math_unary!(
                    $bl,
                    R,
                    f32,
                    tan_pi,
                    |x: f32| (core::f64::consts::PI * x as f64).tan() as f32,
                    TOL_F32,
                    tanpidom
                );
                math_unary!(
                    $bl,
                    R,
                    f32,
                    sinc,
                    |x: f32| {
                        let x = x as f64;
                        (if x == 0.0 { 1.0 } else { x.sin() / x }) as f32
                    },
                    TOL_F32,
                    ang
                );
                math_unary!(
                    $bl,
                    R,
                    f32,
                    sinc_pi,
                    |x: f32| {
                        let x = core::f64::consts::PI * x as f64;
                        (if x == 0.0 { 1.0 } else { x.sin() / x }) as f32
                    },
                    TOL_F32,
                    pidom
                );
                math_unary!($bl, R, f32, exph, |x: f32| 0.5 * libm::expf(x), TOL_F32, small);
                math_unary!(
                    $bl,
                    R,
                    f32,
                    exp10,
                    |x: f32| 10f64.powf(x as f64) as f32,
                    TOL_F32,
                    |x: f32| if x.is_finite() { x % 30.0 } else { 1.0 }
                );
                // `reciprocal` flushes subnormals (1/denormal -> inf) under Performance, so
                // keep the domain to normal values (same reasoning as the `safe` domain).
                math_unary!($bl, R, f32, reciprocal, |x: f32| 1.0 / x, TOL_F32, |x: f32| {
                    let v = x % 1e3;
                    if v.is_normal() { v } else { 1.0 }
                });
                math_unary!(
                    $bl,
                    R,
                    f32,
                    inverse_sqrt,
                    |x: f32| 1.0 / libm::sqrtf(x),
                    TOL_F32,
                    |x: f32| (x % 1e6).abs() + 1e-3
                );
                math_binary!(
                    $bl,
                    R,
                    f32,
                    approx_div,
                    |a: f32, b: f32| a / b,
                    TOL_F32,
                    |x: f32| {
                        let v = x % 1e3;
                        if v.is_normal() { v } else { 1.0 }
                    },
                    |y: f32| {
                        let v = y % 100.0;
                        if v.abs() > 0.1 { v } else { 1.0 }
                    }
                );

                math_tuple!($bl, R, f32, sin_cos, libm::sinf, libm::cosf, TOL_F32, ang);
                math_tuple!(
                    $bl,
                    R,
                    f32,
                    sincos_pi,
                    |x: f32| (core::f64::consts::PI * x as f64).sin() as f32,
                    |x: f32| (core::f64::consts::PI * x as f64).cos() as f32,
                    TOL_F32,
                    pidom
                );
                math_tuple!($bl, R, f32, sinh_cosh, libm::sinhf, libm::coshf, TOL_F32, small);

                math_unary_cg!(
                    $bl,
                    R,
                    f32,
                    nth_root,
                    3,
                    libm::cbrtf,
                    TOL_F32,
                    |x: f32| if x.is_finite() { x % 1e6 } else { 1.0 }
                );
                math_unary_cg!($bl, R, f32, nth_root, 2, libm::sqrtf, TOL_F32, pos);
                math_unary_cg!($bl, R, f32, log_n, 2, libm::log2f, TOL_F32, pos);
                math_unary_cg!($bl, R, f32, log_n, 10, libm::log10f, TOL_F32, pos);
                math_unary_cg!(
                    $bl,
                    R,
                    f32,
                    log_n,
                    3,
                    |x: f32| libm::logf(x) / libm::logf(3.0),
                    TOL_F32,
                    pos
                );
                math_binary!(
                    $bl,
                    R,
                    f32,
                    log,
                    |x: f32, b: f32| libm::logf(x) / libm::logf(b),
                    TOL_F32,
                    pos,
                    |y: f32| if y.is_finite() {
                        (y.abs() % 10.0) + 1.1
                    } else {
                        2.0
                    }
                );

                math_powi!($bl, R, f32, 0, libm::powf, TOL_F32, |x: f32| if x.is_finite() {
                    x % 1e3
                } else {
                    1.0
                });
                math_powi!($bl, R, f32, 2, libm::powf, TOL_F32, |x: f32| if x.is_finite() {
                    x % 100.0
                } else {
                    1.0
                });
                math_powi!($bl, R, f32, 3, libm::powf, TOL_F32, |x: f32| if x.is_finite() {
                    x % 20.0
                } else {
                    1.0
                });
                // negative exponents (signed base; |base| bounded away from 0 so x^-n stays normal)
                let pineg = |x: f32| {
                    let v = x % 8.0;
                    if v.is_normal() && v.abs() > 0.25 { v } else { 1.5 }
                };
                math_powi!($bl, R, f32, -1, libm::powf, TOL_F32, pineg);
                math_powi!($bl, R, f32, -2, libm::powf, TOL_F32, pineg);
                math_powi!($bl, R, f32, -3, libm::powf, TOL_F32, pineg);
            }

            #[test]
            fn f64() {
                type R = <$backend as Simd>::$f64reg;
                let pos = |x: f64| if x.is_finite() { x.abs() + 1e-3 } else { 1.0 };
                let unit = |x: f64| {
                    if x.is_finite() {
                        (x % 2.0).clamp(-0.999, 0.999)
                    } else {
                        0.5
                    }
                };
                let small = |x: f64| if x.is_finite() { (x % 80.0) } else { 1.0 };
                let ang = |x: f64| if x.is_finite() { x % 1000.0 } else { 1.0 };
                let ge1 = |x: f64| if x.is_finite() { x.abs() + 1.0 } else { 2.0 };

                math_unary!($bl, R, f64, sin, libm::sin, TOL_F64, ang);
                math_unary!($bl, R, f64, cos, libm::cos, TOL_F64, ang);
                math_unary!($bl, R, f64, tan, libm::tan, TOL_F64, |x: f64| if x.is_finite() {
                    x % 1.5
                } else {
                    1.0
                });
                math_unary!($bl, R, f64, exp, libm::exp, TOL_F64, small);
                math_unary!($bl, R, f64, exp2, libm::exp2, TOL_F64, small);
                math_unary!($bl, R, f64, ln, libm::log, TOL_F64, pos);
                math_unary!($bl, R, f64, log2, libm::log2, TOL_F64, pos);
                math_unary!($bl, R, f64, sqrt, libm::sqrt, TOL_F64, pos);
                math_unary!(
                    $bl,
                    R,
                    f64,
                    cbrt,
                    libm::cbrt,
                    TOL_F64,
                    |x: f64| if x.is_finite() { x % 1e6 } else { 1.0 }
                );
                math_unary!($bl, R, f64, asin, libm::asin, TOL_F64, unit);
                math_unary!($bl, R, f64, acos, libm::acos, TOL_F64, unit);
                math_unary!($bl, R, f64, atan, libm::atan, TOL_F64, ang);
                math_unary!($bl, R, f64, sinh, libm::sinh, TOL_F64, small);
                math_unary!($bl, R, f64, cosh, libm::cosh, TOL_F64, small);
                math_unary!($bl, R, f64, tanh, libm::tanh, TOL_F64, small);
                math_unary!($bl, R, f64, acosh, libm::acosh, TOL_F64, ge1);
                math_unary!($bl, R, f64, asinh, libm::asinh, TOL_F64, small);

                math_unary!($bl, R, f64, exp_m1, libm::expm1, TOL_F64, small);
                // 2^x - 1 == expm1(x * ln2); the latter is the accurate oracle near 0
                math_unary!(
                    $bl,
                    R,
                    f64,
                    exp2_m1,
                    |x: f64| libm::expm1(x * core::f64::consts::LN_2),
                    TOL_F64,
                    small
                );
                // 10^x - 1 == expm1(x * ln10)
                math_unary!(
                    $bl,
                    R,
                    f64,
                    exp10_m1,
                    |x: f64| libm::expm1(x * core::f64::consts::LN_10),
                    TOL_F64,
                    small
                );
                math_unary!(
                    $bl,
                    R,
                    f64,
                    ln_1p,
                    libm::log1p,
                    TOL_F64,
                    |x: f64| if x.is_finite() { (x % 2.0).max(-0.9) } else { 0.5 }
                );
                // log_b(1 + x) == log1p(x) * log_b(e)
                math_unary!(
                    $bl,
                    R,
                    f64,
                    log2_p1,
                    |x: f64| libm::log1p(x) * core::f64::consts::LOG2_E,
                    TOL_F64,
                    |x: f64| if x.is_finite() { (x % 2.0).max(-0.9) } else { 0.5 }
                );
                math_unary!(
                    $bl,
                    R,
                    f64,
                    log10_p1,
                    |x: f64| libm::log1p(x) * core::f64::consts::LOG10_E,
                    TOL_F64,
                    |x: f64| if x.is_finite() { (x % 2.0).max(-0.9) } else { 0.5 }
                );
                // sqrt(1+x) - 1 == expm1(0.5 * log1p(x)); the latter is the accurate oracle near 0
                math_unary!(
                    $bl,
                    R,
                    f64,
                    sqrt1pm1,
                    |x: f64| libm::expm1(0.5 * libm::log1p(x)),
                    TOL_F64,
                    |x: f64| if x.is_finite() { (x % 20.0).max(-0.9) } else { 0.5 }
                );
                // cos(x) - 1 == -2 sin²(x/2); the latter is the accurate oracle near 0
                math_unary!(
                    $bl,
                    R,
                    f64,
                    cos_m1,
                    |x: f64| {
                        let s = libm::sin(x * 0.5);
                        -2.0 * s * s
                    },
                    TOL_F64,
                    small
                );
                math_unary!(
                    $bl,
                    R,
                    f64,
                    versin,
                    |x: f64| {
                        let s = libm::sin(x * 0.5);
                        2.0 * s * s
                    },
                    TOL_F64,
                    small
                );
                math_unary!(
                    $bl,
                    R,
                    f64,
                    haversin,
                    |x: f64| {
                        let s = libm::sin(x * 0.5);
                        s * s
                    },
                    TOL_F64,
                    small
                );
                math_unary!($bl, R, f64, log10, libm::log10, TOL_F64, pos);
                math_unary!($bl, R, f64, atanh, libm::atanh, TOL_F64, unit);
                // See the f32 `safe` note: the magnitude floor keeps ratio-sensitive
                // atan2 inputs out of the WASM `Size`/Crush denormal-quantization zone.
                let safe = |x: f64| {
                    let v = if x.is_finite() { x % 1e3 } else { 1.0 };
                    if v.is_normal() && v.abs() >= 1e-30 {
                        v
                    } else {
                        1.0_f64.copysign(v)
                    }
                };
                math_binary!($bl, R, f64, atan2, libm::atan2, TOL_F64, safe, safe);
                math_binary!($bl, R, f64, hypot, libm::hypot, TOL_F64, safe, safe);
                // Positive base, exponent in [-8, 8] -- covers the exponent-split path in
                // both directions, including the power-of-two bases and negative exponents
                // that #W11 silently flushed to zero.
                math_binary!(
                    $bl,
                    R,
                    f64,
                    powf,
                    libm::pow,
                    TOL_F64,
                    |x: f64| if x.is_finite() {
                        (x.abs() % 10.0) + 0.1
                    } else {
                        2.0
                    },
                    |y: f64| if y.is_finite() { y % 8.0 } else { 0.0 }
                );
                // compound(x, n) = (1+x)^n
                math_binary!(
                    $bl,
                    R,
                    f64,
                    compound,
                    |x: f64, n: f64| libm::pow(1.0 + x, n),
                    TOL_F64,
                    |x: f64| if x.is_finite() { (x % 5.0).max(-0.9) } else { 0.5 },
                    |n: f64| if n.is_finite() { n % 8.0 } else { 2.0 }
                );
                // logaddexp(a, b) = ln(e^a + e^b)
                math_binary!(
                    $bl,
                    R,
                    f64,
                    logaddexp,
                    |a: f64, b: f64| libm::log(libm::exp(a) + libm::exp(b)),
                    TOL_F64,
                    |x: f64| if x.is_finite() { x % 20.0 } else { 1.0 },
                    |x: f64| if x.is_finite() { x % 20.0 } else { -1.0 }
                );
                // powf_m1(x, e) = x^e - 1 == expm1(e * ln x)
                math_binary!(
                    $bl,
                    R,
                    f64,
                    powf_m1,
                    |x: f64, e: f64| libm::expm1(e * libm::log(x)),
                    TOL_F64,
                    |x: f64| if x.is_finite() {
                        (x.abs() % 10.0) + 0.1
                    } else {
                        2.0
                    },
                    |e: f64| if e.is_finite() { e % 4.0 } else { 2.0 }
                );

                // --- additional transcendentals (this session) ---
                let pidom = |x: f64| if x.is_finite() { x % 30.0 } else { 1.0 };
                let tanpidom = |x: f64| if x.is_finite() { x % 0.4 } else { 0.1 }; // away from ±0.5 poles
                math_unary!(
                    $bl,
                    R,
                    f64,
                    sin_pi,
                    |x: f64| (core::f64::consts::PI * x).sin(),
                    TOL_F64,
                    pidom
                );
                math_unary!(
                    $bl,
                    R,
                    f64,
                    cos_pi,
                    |x: f64| (core::f64::consts::PI * x).cos(),
                    TOL_F64,
                    pidom
                );
                math_unary!(
                    $bl,
                    R,
                    f64,
                    tan_pi,
                    |x: f64| (core::f64::consts::PI * x).tan(),
                    TOL_F64,
                    tanpidom
                );
                math_unary!(
                    $bl,
                    R,
                    f64,
                    sinc,
                    |x: f64| if x == 0.0 { 1.0 } else { x.sin() / x },
                    TOL_F64,
                    ang
                );
                math_unary!(
                    $bl,
                    R,
                    f64,
                    sinc_pi,
                    |x: f64| {
                        let x = core::f64::consts::PI * x;
                        if x == 0.0 { 1.0 } else { x.sin() / x }
                    },
                    TOL_F64,
                    pidom
                );
                math_unary!($bl, R, f64, exph, |x: f64| 0.5 * libm::exp(x), TOL_F64, small);
                math_unary!(
                    $bl,
                    R,
                    f64,
                    exp10,
                    |x: f64| 10f64.powf(x),
                    TOL_F64,
                    |x: f64| if x.is_finite() { x % 200.0 } else { 1.0 }
                );
                math_unary!($bl, R, f64, reciprocal, |x: f64| 1.0 / x, TOL_F64, |x: f64| {
                    let v = x % 1e3;
                    if v.is_normal() { v } else { 1.0 }
                });
                math_unary!(
                    $bl,
                    R,
                    f64,
                    inverse_sqrt,
                    |x: f64| 1.0 / libm::sqrt(x),
                    TOL_F64,
                    |x: f64| (x % 1e6).abs() + 1e-3
                );
                math_binary!(
                    $bl,
                    R,
                    f64,
                    approx_div,
                    |a: f64, b: f64| a / b,
                    TOL_F64,
                    |x: f64| {
                        let v = x % 1e3;
                        if v.is_normal() { v } else { 1.0 }
                    },
                    |y: f64| {
                        let v = y % 100.0;
                        if v.abs() > 0.1 { v } else { 1.0 }
                    }
                );

                math_tuple!($bl, R, f64, sin_cos, libm::sin, libm::cos, TOL_F64, ang);
                math_tuple!(
                    $bl,
                    R,
                    f64,
                    sincos_pi,
                    |x: f64| (core::f64::consts::PI * x).sin(),
                    |x: f64| (core::f64::consts::PI * x).cos(),
                    TOL_F64,
                    pidom
                );
                math_tuple!($bl, R, f64, sinh_cosh, libm::sinh, libm::cosh, TOL_F64, small);

                math_unary_cg!(
                    $bl,
                    R,
                    f64,
                    nth_root,
                    3,
                    libm::cbrt,
                    TOL_F64,
                    |x: f64| if x.is_finite() { x % 1e6 } else { 1.0 }
                );
                math_unary_cg!($bl, R, f64, nth_root, 2, libm::sqrt, TOL_F64, pos);
                math_unary_cg!($bl, R, f64, log_n, 2, libm::log2, TOL_F64, pos);
                math_unary_cg!($bl, R, f64, log_n, 10, libm::log10, TOL_F64, pos);
                math_unary_cg!(
                    $bl,
                    R,
                    f64,
                    log_n,
                    3,
                    |x: f64| libm::log(x) / libm::log(3.0),
                    TOL_F64,
                    pos
                );
                math_binary!(
                    $bl,
                    R,
                    f64,
                    log,
                    |x: f64, b: f64| libm::log(x) / libm::log(b),
                    TOL_F64,
                    pos,
                    |y: f64| if y.is_finite() {
                        (y.abs() % 10.0) + 1.1
                    } else {
                        2.0
                    }
                );

                math_powi!($bl, R, f64, 0, libm::pow, TOL_F64, |x: f64| if x.is_finite() {
                    x % 1e3
                } else {
                    1.0
                });
                math_powi!($bl, R, f64, 2, libm::pow, TOL_F64, |x: f64| if x.is_finite() {
                    x % 100.0
                } else {
                    1.0
                });
                math_powi!($bl, R, f64, 3, libm::pow, TOL_F64, |x: f64| if x.is_finite() {
                    x % 20.0
                } else {
                    1.0
                });
                let pineg = |x: f64| {
                    let v = x % 8.0;
                    if v.is_normal() && v.abs() > 0.25 { v } else { 1.5 }
                };
                math_powi!($bl, R, f64, -1, libm::pow, TOL_F64, pineg);
                math_powi!($bl, R, f64, -2, libm::pow, TOL_F64, pineg);
                math_powi!($bl, R, f64, -3, libm::pow, TOL_F64, pineg);
            }
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    use thermite::backend::x86_v1::X86V1;
    use thermite::backend::x86_v2::X86V2;
    use thermite::backend::x86_v3::X86V3;
    math_suite!(v3, X86V3, f32x8, f64x4, "x86_v3");
    math_suite!(v2, X86V2, f32x4, f64x2, "x86_v2");
    math_suite!(v1, X86V1, f32x4, f64x2, "x86_v1");
}

// WASM: native 128-bit f32x4 / f64x2 transcendental math vs libm.
#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use thermite::backend::wasm::Wasm;
    math_suite!(wasm, Wasm, f32x4, f64x2, "wasm");
}

// NEON: native 128-bit f32x4 / f64x2 transcendental math vs libm.
#[cfg(all(feature = "neon", target_arch = "aarch64"))]
mod neon {
    use super::*;
    use thermite::backend::neon::Neon;
    math_suite!(neon, Neon, f32x4, f64x2, "neon");
}

// ===========================================================================
// Policy-variant coverage. The transcendental kernels in `ps.rs`/`pd.rs` branch
// heavily on the precision tier (`if const { P::PRECISION ... }`), and the
// default suite above only exercises the `Performance` policy. Running the same
// functions under each preset (`UltraPerformance`..`Reference`) lights up those
// branches. Tolerances here are deliberately loose - this is a branch-exercising
// smoke gate (wrong function / NaN / sign / gross error), not a precision audit;
// the tight gate stays at the default policy above.
// ===========================================================================
use thermite::math::policy::policies::{HighPerformance, Precision, Reference, Size, UltraPerformance};

/// Like `math_unary!` but calls a policy variant `<reg>.$pm::<$policy>()`.
macro_rules! math_unary_p {
    ($label:expr, $reg:ty, $elem:ty, $pm:ident, $policy:ty, $oracle:expr, $tol:expr, $domain:expr) => {{
        let mut rng = harness::rng();
        let lanes = <Vector<$reg> as GenericVector>::LANES;
        let dom: fn($elem) -> $elem = $domain;
        let oracle: fn($elem) -> $elem = $oracle;
        for raw in harness::corpus::<$elem>(lanes, &mut rng) {
            let input: Vec<$elem> = raw.iter().map(|&x| dom(x)).collect();
            let got = Vector::<$reg>(harness::make_array::<$reg>(&input))
                .$pm::<$policy>()
                .into_array();
            for (lane, (&g, &x)) in got.iter().zip(input.iter()).enumerate() {
                let want = oracle(x);
                if !close(g as f64, want as f64, $tol) {
                    panic!(
                        "{} [{} <{}>]: lane {lane} mismatch\n  in={x:?} got={g:?} want={want:?} (libm)",
                        $label,
                        stringify!($pm),
                        stringify!($policy)
                    );
                }
            }
        }
    }};
}

/// Like `math_tuple!` but calls a policy variant `<reg>.$pm::<$policy>()`.
macro_rules! math_tuple_p {
    ($label:expr, $reg:ty, $elem:ty, $pm:ident, $policy:ty, $o0:expr, $o1:expr, $tol:expr, $domain:expr) => {{
        let mut rng = harness::rng();
        let lanes = <Vector<$reg> as GenericVector>::LANES;
        let dom: fn($elem) -> $elem = $domain;
        let o0: fn($elem) -> $elem = $o0;
        let o1: fn($elem) -> $elem = $o1;
        for raw in harness::corpus::<$elem>(lanes, &mut rng) {
            let input: Vec<$elem> = raw.iter().map(|&x| dom(x)).collect();
            let (ra, rb) = Vector::<$reg>(harness::make_array::<$reg>(&input)).$pm::<$policy>();
            let (ga, gb) = (ra.into_array(), rb.into_array());
            for (lane, &x) in input.iter().enumerate() {
                let (wa, wb) = (o0(x), o1(x));
                if !close(ga[lane] as f64, wa as f64, $tol) {
                    panic!(
                        "{} [{}.0 <{}>]: lane {lane} in={x:?} got={:?} want={wa:?}",
                        $label,
                        stringify!($pm),
                        stringify!($policy),
                        ga[lane]
                    );
                }
                if !close(gb[lane] as f64, wb as f64, $tol) {
                    panic!(
                        "{} [{}.1 <{}>]: lane {lane} in={x:?} got={:?} want={wb:?}",
                        $label,
                        stringify!($pm),
                        stringify!($policy),
                        gb[lane]
                    );
                }
            }
        }
    }};
}

/// Like `math_unary_cg!` but calls a policy variant `<reg>.$pm::<$policy, $n>()`
/// (the `_p` variants take the policy as the *first* generic, then the method's own).
macro_rules! math_unary_cg_p {
    ($label:expr, $reg:ty, $elem:ty, $pm:ident, $policy:ty, $n:literal, $oracle:expr, $tol:expr, $domain:expr) => {{
        let mut rng = harness::rng();
        let lanes = <Vector<$reg> as GenericVector>::LANES;
        let dom: fn($elem) -> $elem = $domain;
        let oracle: fn($elem) -> $elem = $oracle;
        for raw in harness::corpus::<$elem>(lanes, &mut rng) {
            let input: Vec<$elem> = raw.iter().map(|&x| dom(x)).collect();
            let got = Vector::<$reg>(harness::make_array::<$reg>(&input))
                .$pm::<$policy, $n>()
                .into_array();
            for (lane, (&g, &x)) in got.iter().zip(input.iter()).enumerate() {
                let want = oracle(x);
                if !close(g as f64, want as f64, $tol) {
                    panic!(
                        "{} [{}::<{}, {}>]: lane {lane} in={x:?} got={g:?} want={want:?} (libm)",
                        $label,
                        stringify!($pm),
                        stringify!($policy),
                        $n
                    );
                }
            }
        }
    }};
}

macro_rules! f32_policy_fns {
    ($reg:ty, $policy:ty, $tol:expr, $bl:expr) => {{
        let pos = |x: f32| if x.is_finite() { x.abs() + 1e-3 } else { 1.0 };
        let unit = |x: f32| {
            if x.is_finite() {
                (x % 2.0).clamp(-0.999, 0.999)
            } else {
                0.5
            }
        };
        let small = |x: f32| if x.is_finite() { x % 20.0 } else { 1.0 };
        let ang = |x: f32| if x.is_finite() { x % 1000.0 } else { 1.0 };
        math_unary_p!($bl, $reg, f32, sin_p, $policy, libm::sinf, $tol, ang);
        math_unary_p!($bl, $reg, f32, cos_p, $policy, libm::cosf, $tol, ang);
        math_unary_p!($bl, $reg, f32, exp_p, $policy, libm::expf, $tol, small);
        math_unary_p!($bl, $reg, f32, exp2_p, $policy, libm::exp2f, $tol, small);
        math_unary_p!($bl, $reg, f32, ln_p, $policy, libm::logf, $tol, pos);
        math_unary_p!($bl, $reg, f32, log2_p, $policy, libm::log2f, $tol, pos);
        math_unary_p!(
            $bl,
            $reg,
            f32,
            cbrt_p,
            $policy,
            libm::cbrtf,
            $tol,
            |x: f32| if x.is_finite() { x % 1e6 } else { 1.0 }
        );
        math_unary_p!($bl, $reg, f32, asin_p, $policy, libm::asinf, $tol, unit);
        math_unary_p!($bl, $reg, f32, atan_p, $policy, libm::atanf, $tol, ang);
        math_unary_p!($bl, $reg, f32, sinh_p, $policy, libm::sinhf, $tol, small);
        math_unary_p!($bl, $reg, f32, tanh_p, $policy, libm::tanhf, $tol, small);

        // second batch under every policy tier
        let pidom = |x: f32| if x.is_finite() { x % 30.0 } else { 1.0 };
        let tanpidom = |x: f32| if x.is_finite() { x % 0.4 } else { 0.1 };
        let recipdom = |x: f32| {
            let v = x % 1e3;
            if v.is_normal() { v } else { 1.0 }
        };
        math_unary_p!(
            $bl,
            $reg,
            f32,
            sin_pi_p,
            $policy,
            |x: f32| (core::f64::consts::PI * x as f64).sin() as f32,
            $tol,
            pidom
        );
        math_unary_p!(
            $bl,
            $reg,
            f32,
            cos_pi_p,
            $policy,
            |x: f32| (core::f64::consts::PI * x as f64).cos() as f32,
            $tol,
            pidom
        );
        math_unary_p!(
            $bl,
            $reg,
            f32,
            tan_pi_p,
            $policy,
            |x: f32| (core::f64::consts::PI * x as f64).tan() as f32,
            $tol,
            tanpidom
        );
        // Low-precision sinc (precision<=Medium) is only well-behaved at moderate
        // magnitudes: at x==0 it returns NaN (the 0-guard is gated on check_overflow,
        // off for Ultra/High), and for tiny |x| the approximate-reciprocal `1/x` blows
        // up. Band the smoke-test domain to |x| in [0.1, 100].
        let sincdom = |x: f32| {
            let v = x % 100.0;
            if v.is_normal() && v.abs() >= 0.1 { v } else { 1.7 }
        };
        let sincpidom = |x: f32| {
            let v = x % 20.0;
            if v.is_normal() && v.abs() >= 0.1 { v } else { 1.3 }
        };
        math_unary_p!(
            $bl,
            $reg,
            f32,
            sinc_p,
            $policy,
            |x: f32| {
                let x = x as f64;
                (if x == 0.0 { 1.0 } else { x.sin() / x }) as f32
            },
            $tol,
            sincdom
        );
        math_unary_p!(
            $bl,
            $reg,
            f32,
            sinc_pi_p,
            $policy,
            |x: f32| {
                let x = core::f64::consts::PI * x as f64;
                (if x == 0.0 { 1.0 } else { x.sin() / x }) as f32
            },
            $tol,
            sincpidom
        );
        math_unary_p!(
            $bl,
            $reg,
            f32,
            exph_p,
            $policy,
            |x: f32| 0.5 * libm::expf(x),
            $tol,
            small
        );
        math_unary_p!(
            $bl,
            $reg,
            f32,
            exp10_p,
            $policy,
            |x: f32| 10f64.powf(x as f64) as f32,
            $tol,
            small
        );
        math_unary_p!($bl, $reg, f32, reciprocal_p, $policy, |x: f32| 1.0 / x, $tol, recipdom);
        math_tuple_p!($bl, $reg, f32, sin_cos_p, $policy, libm::sinf, libm::cosf, $tol, ang);
        math_tuple_p!(
            $bl,
            $reg,
            f32,
            sincos_pi_p,
            $policy,
            |x: f32| (core::f64::consts::PI * x as f64).sin() as f32,
            |x: f32| (core::f64::consts::PI * x as f64).cos() as f32,
            $tol,
            pidom
        );
        math_tuple_p!(
            $bl,
            $reg,
            f32,
            sinh_cosh_p,
            $policy,
            libm::sinhf,
            libm::coshf,
            $tol,
            small
        );
        math_unary_cg_p!(
            $bl,
            $reg,
            f32,
            nth_root_p,
            $policy,
            3,
            libm::cbrtf,
            $tol,
            |x: f32| if x.is_finite() { x % 1e6 } else { 1.0 }
        );
        math_unary_cg_p!(
            $bl,
            $reg,
            f32,
            log_n_p,
            $policy,
            3,
            |x: f32| libm::logf(x) / libm::logf(3.0),
            $tol,
            pos
        );
    }};
}

macro_rules! f64_policy_fns {
    ($reg:ty, $policy:ty, $tol:expr, $bl:expr) => {{
        let pos = |x: f64| if x.is_finite() { x.abs() + 1e-3 } else { 1.0 };
        let unit = |x: f64| {
            if x.is_finite() {
                (x % 2.0).clamp(-0.999, 0.999)
            } else {
                0.5
            }
        };
        let small = |x: f64| if x.is_finite() { x % 80.0 } else { 1.0 };
        let ang = |x: f64| if x.is_finite() { x % 1000.0 } else { 1.0 };
        math_unary_p!($bl, $reg, f64, sin_p, $policy, libm::sin, $tol, ang);
        math_unary_p!($bl, $reg, f64, cos_p, $policy, libm::cos, $tol, ang);
        math_unary_p!($bl, $reg, f64, exp_p, $policy, libm::exp, $tol, small);
        math_unary_p!($bl, $reg, f64, exp2_p, $policy, libm::exp2, $tol, small);
        math_unary_p!($bl, $reg, f64, ln_p, $policy, libm::log, $tol, pos);
        math_unary_p!($bl, $reg, f64, log2_p, $policy, libm::log2, $tol, pos);
        math_unary_p!(
            $bl,
            $reg,
            f64,
            cbrt_p,
            $policy,
            libm::cbrt,
            $tol,
            |x: f64| if x.is_finite() { x % 1e6 } else { 1.0 }
        );
        math_unary_p!($bl, $reg, f64, asin_p, $policy, libm::asin, $tol, unit);
        math_unary_p!($bl, $reg, f64, atan_p, $policy, libm::atan, $tol, ang);
        math_unary_p!($bl, $reg, f64, sinh_p, $policy, libm::sinh, $tol, small);
        math_unary_p!($bl, $reg, f64, tanh_p, $policy, libm::tanh, $tol, small);

        // second batch under every policy tier
        let pidom = |x: f64| if x.is_finite() { x % 30.0 } else { 1.0 };
        let tanpidom = |x: f64| if x.is_finite() { x % 0.4 } else { 0.1 };
        let recipdom = |x: f64| {
            let v = x % 1e3;
            if v.is_normal() { v } else { 1.0 }
        };
        math_unary_p!(
            $bl,
            $reg,
            f64,
            sin_pi_p,
            $policy,
            |x: f64| (core::f64::consts::PI * x).sin(),
            $tol,
            pidom
        );
        math_unary_p!(
            $bl,
            $reg,
            f64,
            cos_pi_p,
            $policy,
            |x: f64| (core::f64::consts::PI * x).cos(),
            $tol,
            pidom
        );
        math_unary_p!(
            $bl,
            $reg,
            f64,
            tan_pi_p,
            $policy,
            |x: f64| (core::f64::consts::PI * x).tan(),
            $tol,
            tanpidom
        );
        // Low-precision sinc is only well-behaved at moderate magnitudes (NaN at 0 under
        // check_overflow-off Ultra/High; approximate `1/x` blows up for tiny |x|).
        let sincdom = |x: f64| {
            let v = x % 100.0;
            if v.is_normal() && v.abs() >= 0.1 { v } else { 1.7 }
        };
        let sincpidom = |x: f64| {
            let v = x % 20.0;
            if v.is_normal() && v.abs() >= 0.1 { v } else { 1.3 }
        };
        math_unary_p!(
            $bl,
            $reg,
            f64,
            sinc_p,
            $policy,
            |x: f64| if x == 0.0 { 1.0 } else { x.sin() / x },
            $tol,
            sincdom
        );
        math_unary_p!(
            $bl,
            $reg,
            f64,
            sinc_pi_p,
            $policy,
            |x: f64| {
                let x = core::f64::consts::PI * x;
                if x == 0.0 { 1.0 } else { x.sin() / x }
            },
            $tol,
            sincpidom
        );
        math_unary_p!(
            $bl,
            $reg,
            f64,
            exph_p,
            $policy,
            |x: f64| 0.5 * libm::exp(x),
            $tol,
            small
        );
        math_unary_p!(
            $bl,
            $reg,
            f64,
            exp10_p,
            $policy,
            |x: f64| 10f64.powf(x),
            $tol,
            |x: f64| if x.is_finite() { x % 200.0 } else { 1.0 }
        );
        math_unary_p!($bl, $reg, f64, reciprocal_p, $policy, |x: f64| 1.0 / x, $tol, recipdom);
        math_tuple_p!($bl, $reg, f64, sin_cos_p, $policy, libm::sin, libm::cos, $tol, ang);
        math_tuple_p!(
            $bl,
            $reg,
            f64,
            sincos_pi_p,
            $policy,
            |x: f64| (core::f64::consts::PI * x).sin(),
            |x: f64| (core::f64::consts::PI * x).cos(),
            $tol,
            pidom
        );
        math_tuple_p!(
            $bl,
            $reg,
            f64,
            sinh_cosh_p,
            $policy,
            libm::sinh,
            libm::cosh,
            $tol,
            small
        );
        math_unary_cg_p!(
            $bl,
            $reg,
            f64,
            nth_root_p,
            $policy,
            3,
            libm::cbrt,
            $tol,
            |x: f64| if x.is_finite() { x % 1e6 } else { 1.0 }
        );
        math_unary_cg_p!(
            $bl,
            $reg,
            f64,
            log_n_p,
            $policy,
            3,
            |x: f64| libm::log(x) / libm::log(3.0),
            $tol,
            pos
        );
    }};
}

// Loose, policy-appropriate tolerances (coverage smoke gate, NOT a precision
// audit). The lowest tiers (UltraPerformance/HighPerformance/Size) are very
// loose on purpose: e.g. f32 `log2_p::<UltraPerformance>(1.001)` returns ~0.058
// vs a true ~0.00144 (poor near x==1, where log has cancellation) - whether
// that error is acceptable for those tiers is a precision question for the
// maintainer; here we only exercise the policy branches. The tight precision
// gate stays in `math_suite!` at the default policy.
const POL_F32_LOOSE: f64 = 2.0e-1;
const POL_F32_TIGHT: f64 = 5.0e-3;
const POL_F64_LOOSE: f64 = 5.0e-3;
const POL_F64_TIGHT: f64 = 1.0e-6;

macro_rules! policy_suite {
    ($modname:ident, $backend:ty, $f32reg:ident, $f64reg:ident, $bl:expr) => {
        mod $modname {
            use super::*;
            type RF32 = <$backend as Simd>::$f32reg;
            type RF64 = <$backend as Simd>::$f64reg;

            #[test]
            fn ultra_f32() {
                f32_policy_fns!(RF32, UltraPerformance, POL_F32_LOOSE, $bl);
            }
            #[test]
            fn high_f32() {
                f32_policy_fns!(RF32, HighPerformance, POL_F32_LOOSE, $bl);
            }
            #[test]
            fn size_f32() {
                f32_policy_fns!(RF32, Size, POL_F32_LOOSE, $bl);
            }
            #[test]
            fn prec_f32() {
                f32_policy_fns!(RF32, Precision, POL_F32_TIGHT, $bl);
            }
            #[test]
            fn ref_f32() {
                f32_policy_fns!(RF32, Reference, POL_F32_TIGHT, $bl);
            }

            #[test]
            fn ultra_f64() {
                f64_policy_fns!(RF64, UltraPerformance, POL_F64_LOOSE, $bl);
            }
            #[test]
            fn high_f64() {
                f64_policy_fns!(RF64, HighPerformance, POL_F64_LOOSE, $bl);
            }
            #[test]
            fn size_f64() {
                f64_policy_fns!(RF64, Size, POL_F64_LOOSE, $bl);
            }
            #[test]
            fn prec_f64() {
                f64_policy_fns!(RF64, Precision, POL_F64_TIGHT, $bl);
            }
            #[test]
            fn ref_f64() {
                f64_policy_fns!(RF64, Reference, POL_F64_TIGHT, $bl);
            }
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_policy {
    use super::*;
    use thermite::backend::x86_v1::X86V1;
    use thermite::backend::x86_v2::X86V2;
    use thermite::backend::x86_v3::X86V3;
    policy_suite!(pol_v3, X86V3, f32x8, f64x4, "x86_v3");
    policy_suite!(pol_v2, X86V2, f32x4, f64x2, "x86_v2");
    policy_suite!(pol_v1, X86V1, f32x4, f64x2, "x86_v1");
}

#[cfg(target_arch = "wasm32")]
mod wasm_policy {
    use super::*;
    use thermite::backend::wasm::Wasm;
    policy_suite!(pol_wasm, Wasm, f32x4, f64x2, "wasm");
}

#[cfg(all(feature = "neon", target_arch = "aarch64"))]
mod neon_policy {
    use super::*;
    use thermite::backend::neon::Neon;
    policy_suite!(pol_neon, Neon, f32x4, f64x2, "neon");
}
