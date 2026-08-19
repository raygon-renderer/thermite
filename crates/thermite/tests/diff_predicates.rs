//! Float / signed predicate coverage: the mask-returning classification
//! methods (`is_nan`, `is_finite`, `is_infinite`, `is_normal`, `is_subnormal`,
//! `is_zero_or_subnormal`, `is_negative`, `is_positive`) plus integer `signum`.
//!
//! Each is checked against the equivalent Rust method / operator over the full
//! edge-case + random corpus, on `Scalar`, `X86V2`, and `X86V3`.
//!
//! Note `is_nan` is implemented as `ne(value, value)`, so this is also the
//! regression gate for the V3 `ne` fix (ordered `_CMP_NEQ_OQ` -> unordered
//! `_CMP_NEQ_UQ`): before it, `is_nan(NaN)` was `false` on f32x8/f64x{2,4}.
//!
//! Semantics oracles match thermite's *comparison-based* definitions, which
//! differ from the sign-bit ones for ±0.0:
//!   - `is_negative(x) == (x < 0)`   -> `is_negative(-0.0)` is `false`
//!   - `is_positive(x) == (x >= 0)`  -> `is_positive(-0.0)` is `true`
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    target_arch = "aarch64"
))]

mod harness;

use generic_array::typenum::Unsigned;

use harness::Tol;
use thermite::register::{CoreRegister, FloatRegister as _, Register, SignedRegister as _};
use thermite::simd::Simd;

use thermite::backend::scalar::Scalar;

/// A unary predicate `Storage<Self> -> Storage<Self::Mask>` vs a Rust
/// `Fn($elem) -> bool` oracle.
macro_rules! pred {
    ($label:expr, $ut:ty, $e:ty, $method:ident, $op:expr) => {{
        let mut rng = harness::rng();
        let lanes = <<$ut as CoreRegister>::Lanes as Unsigned>::USIZE;
        let oracle: fn($e) -> bool = $op;
        for x in harness::corpus::<$e>(lanes, &mut rng) {
            let m = <$ut>::$method(harness::make_array::<$ut>(&x));
            let got = harness::read_mask::<$ut>(m, lanes);
            for (lane, &g) in got.iter().enumerate() {
                let w = oracle(x[lane]);
                assert!(
                    g == w,
                    concat!(
                        $label,
                        " [",
                        stringify!($method),
                        "]: lane {} mismatch\n  x = {:?}\n  got = {}  want = {}"
                    ),
                    lane,
                    x[lane],
                    g,
                    w
                );
            }
        }
    }};
}

/// A unary value op `Storage<Self> -> Storage<Self>` vs a Rust `Fn($e) -> $e`
/// oracle (used for `signum`, which returns a vector, not a mask).
macro_rules! val_un {
    ($label:expr, $ut:ty, $e:ty, $method:ident, $op:expr) => {{
        let mut rng = harness::rng();
        let lanes = <<$ut as CoreRegister>::Lanes as Unsigned>::USIZE;
        let oracle: fn($e) -> $e = $op;
        for x in harness::corpus::<$e>(lanes, &mut rng) {
            let got = harness::read::<$ut>(&<$ut>::$method(harness::make_array::<$ut>(&x)));
            let want: Vec<$e> = x.iter().map(|&v| oracle(v)).collect();
            harness::assert_lanes_eq(
                concat!($label, " [", stringify!($method), "]"),
                &[x.as_slice()],
                &got,
                &want,
                Tol::Exact,
            );
        }
    }};
}

/// Float `signum`, NaN-aware. For every non-NaN input it must match `x.signum()`
/// exactly on all backends (finite, ±0 and ±inf all agree, being copysign-based ±1).
/// For a NaN input the result is **unspecified without the `strict_ieee754`
/// feature**: the scalar backend propagates NaN (`f32::signum`), while the x86
/// backends take the fast copysign path and yield ±1 (the NaN blend is gated
/// behind `strict_ieee754`). So a NaN input is allowed to produce NaN *or* ±1.
macro_rules! signum_float {
    ($label:expr, $ut:ty, $e:ty) => {{
        let mut rng = harness::rng();
        let lanes = <<$ut as CoreRegister>::Lanes as Unsigned>::USIZE;
        for x in harness::corpus::<$e>(lanes, &mut rng) {
            let got = harness::read::<$ut>(&<$ut>::signum(harness::make_array::<$ut>(&x)));
            for (i, (&xi, &g)) in x.iter().zip(&got).enumerate() {
                if xi.is_nan() {
                    assert!(
                        g.is_nan() || g.abs() == 1.0,
                        "{} [signum] NaN lane {}: got {}",
                        $label,
                        i,
                        g
                    );
                } else {
                    assert_eq!(g, xi.signum(), "{} [signum] lane {}: in {}", $label, i, xi);
                }
            }
        }
    }};
}

macro_rules! float_pred_tests {
    ($name:ident, $backend:ty, $reg:ident, $label:expr) => {
        #[test]
        fn $name() {
            type UT = <$backend as Simd>::$reg;
            type E = <UT as Register>::Element;
            pred!($label, UT, E, is_nan, |x| x.is_nan());
            pred!($label, UT, E, is_infinite, |x| x.is_infinite());
            pred!($label, UT, E, is_finite, |x| x.is_finite());
            pred!($label, UT, E, is_normal, |x| x.is_normal());
            pred!($label, UT, E, is_subnormal, |x| x.is_subnormal());
            pred!($label, UT, E, is_zero_or_subnormal, |x| x == 0.0
                || x.is_subnormal());
            pred!($label, UT, E, is_negative, |x| x < 0.0);
            pred!($label, UT, E, is_positive, |x| x >= 0.0);
            // Sign-bit ±1 for every non-NaN input (signum(+0)=+1, signum(-0)=-1,
            // signum(±inf)=±1), matching Rust `f32::signum`. NaN signum is only
            // NaN under `strict_ieee754`. By default x86 yields ±1, see `signum_float`.
            signum_float!($label, UT, E);
        }
    };
}

macro_rules! int_pred_tests {
    ($name:ident, $backend:ty, $reg:ident, $label:expr) => {
        #[test]
        fn $name() {
            type UT = <$backend as Simd>::$reg;
            type E = <UT as Register>::Element;
            pred!($label, UT, E, is_negative, |x| x < 0);
            pred!($label, UT, E, is_positive, |x| x >= 0);
            // Matches Rust `i32::signum`: three-valued -1 / 0 / +1. The x86
            // overrides were fixed (were two-valued, returning +1 for 0).
            val_un!($label, UT, E, signum, |x: E| x.signum());
        }
    };
}

macro_rules! float_suite {
    ($modname:ident, $backend:ty, [$($reg:ident),*], $bl:expr) => {
        mod $modname {
            use super::*;
            $( float_pred_tests!($reg, $backend, $reg, concat!($bl, " ", stringify!($reg))); )*
        }
    };
}
macro_rules! int_suite {
    ($modname:ident, $backend:ty, [$($reg:ident),*], $bl:expr) => {
        mod $modname {
            use super::*;
            $( int_pred_tests!($reg, $backend, $reg, concat!($bl, " ", stringify!($reg))); )*
        }
    };
}

float_suite!(scalar_float, Scalar, [f32x4, f32x8, f64x2, f64x4], "scalar");
int_suite!(scalar_int, Scalar, [i32x4, i32x8, i64x2, i64x4], "scalar");

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    use thermite::backend::x86_v1::X86V1;
    use thermite::backend::x86_v2::X86V2;
    use thermite::backend::x86_v3::X86V3;
    float_suite!(v3_float, X86V3, [f32x4, f32x8, f32x16, f64x2, f64x4, f64x8], "x86_v3");
    float_suite!(v2_float, X86V2, [f32x4, f32x8, f64x2, f64x4], "x86_v2");
    float_suite!(v1_float, X86V1, [f32x4, f32x8, f64x2, f64x4], "x86_v1");

    int_suite!(v3_int, X86V3, [i32x4, i32x8, i64x2, i64x4], "x86_v3");
    int_suite!(v2_int, X86V2, [i32x4, i32x8, i64x2], "x86_v2");
    int_suite!(v1_int, X86V1, [i32x4, i32x8, i64x2], "x86_v1");
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use thermite::backend::wasm::Wasm;
    float_suite!(wasm_float, Wasm, [f32x4, f32x8, f64x2, f64x4], "wasm");
    int_suite!(wasm_int, Wasm, [i32x4, i32x8, i64x2, i64x4], "wasm");
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use thermite::backend::neon::Neon;
    float_suite!(neon_float, Neon, [f32x4, f32x8, f64x2, f64x4], "neon");
    int_suite!(neon_int, Neon, [i32x4, i32x8, i64x2, i64x4], "neon");
}
