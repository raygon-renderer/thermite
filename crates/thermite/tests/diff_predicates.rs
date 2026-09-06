//! Float / signed predicate coverage: the mask-returning classification
//! methods (`is_nan`, `is_finite`, `is_infinite`, `is_normal`, `is_subnormal`,
//! `is_zero_or_subnormal`, `is_negative`, `is_positive`) plus integer `signum`.
//!
//! Each is checked against the equivalent Rust method / operator over the full
//! edge-case + random corpus, on every backend.
//!
//! Note `is_nan` is implemented as `ne(value, value)`, so this is also the
//! regression gate for the V3 `ne` fix (ordered `_CMP_NEQ_OQ` -> unordered
//! `_CMP_NEQ_UQ`): before it, `is_nan(NaN)` was `false` on f32x8/f64x{2,4}.
//!
//! Semantics oracles match thermite's *comparison-based* definitions, which
//! differ from the sign-bit ones for +-0.0:
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
use thermite::register::{CoreRegister, FloatRegister as _, SignedRegister as _};
use thermite::simd::Simd;

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
                    "{} [{}]: lane {} mismatch\n  x = {:?}\n  got = {}  want = {}",
                    $label,
                    stringify!($method),
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
                &format!("{} [{}]", $label, stringify!($method)),
                &[x.as_slice()],
                &got,
                &want,
                Tol::Exact,
            );
        }
    }};
}

/// Float `signum`, NaN-aware. For every non-NaN input it must match `x.signum()`
/// exactly on all backends (finite, +-0 and +-inf all agree, being copysign-based +-1).
/// For a NaN input the result is **unspecified without the `strict_ieee754`
/// feature**: the scalar backend propagates NaN (`f32::signum`), while the x86
/// backends take the fast copysign path and yield +-1 (the NaN blend is gated
/// behind `strict_ieee754`). So a NaN input is allowed to produce NaN _or_ +-1.
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

macro_rules! float_preds {
    ($S:ty, $reg:ident, $e:ty) => {{
        let label = harness::label::<$S>(stringify!($reg));
        let label = label.as_str();
        pred!(label, <$S as Simd>::$reg, $e, is_nan, |x| x.is_nan());
        pred!(label, <$S as Simd>::$reg, $e, is_infinite, |x| x.is_infinite());
        pred!(label, <$S as Simd>::$reg, $e, is_finite, |x| x.is_finite());
        pred!(label, <$S as Simd>::$reg, $e, is_normal, |x| x.is_normal());
        pred!(label, <$S as Simd>::$reg, $e, is_subnormal, |x| x.is_subnormal());
        pred!(label, <$S as Simd>::$reg, $e, is_zero_or_subnormal, |x| x == 0.0 || x.is_subnormal());
        pred!(label, <$S as Simd>::$reg, $e, is_negative, |x| x < 0.0);
        pred!(label, <$S as Simd>::$reg, $e, is_positive, |x| x >= 0.0);
        signum_float!(label, <$S as Simd>::$reg, $e);
    }};
}

macro_rules! int_preds {
    ($S:ty, $reg:ident, $e:ty) => {{
        let label = harness::label::<$S>(stringify!($reg));
        let label = label.as_str();
        pred!(label, <$S as Simd>::$reg, $e, is_negative, |x| x < 0);
        pred!(label, <$S as Simd>::$reg, $e, is_positive, |x| x >= 0);
        val_un!(label, <$S as Simd>::$reg, $e, signum, |x: $e| x.signum());
    }};
}

for_each_backend! {
    fn f32x4<S: Simd>() { float_preds!(S, f32x4, f32) }
    fn f32x8<S: Simd>() { float_preds!(S, f32x8, f32) }
    fn f32x16<S: Simd>() { float_preds!(S, f32x16, f32) }
    fn f64x2<S: Simd>() { float_preds!(S, f64x2, f64) }
    fn f64x4<S: Simd>() { float_preds!(S, f64x4, f64) }
    fn f64x8<S: Simd>() { float_preds!(S, f64x8, f64) }

    fn i32x4<S: Simd>() { int_preds!(S, i32x4, i32) }
    fn i32x8<S: Simd>() { int_preds!(S, i32x8, i32) }
    fn i32x16<S: Simd>() { int_preds!(S, i32x16, i32) }
    fn i64x2<S: Simd>() { int_preds!(S, i64x2, i64) }
    fn i64x4<S: Simd>() { int_preds!(S, i64x4, i64) }
    fn i64x8<S: Simd>() { int_preds!(S, i64x8, i64) }
}
