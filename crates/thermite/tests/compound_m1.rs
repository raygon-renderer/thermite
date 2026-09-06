//! `compound_m1(x, n) = (1 + x)^n - 1`, and the domain-edge regression it shares with
//! `compound` and `powf_m1`.
//!
//! Two things are being pinned. The first is that `compound_m1` is accurate where *both*
//! obvious spellings are not: `compound(x, n) - 1` cancels when the result is near zero, and
//! `powf_m1(1 + x, n)` has thrown `x` away before it starts when `|x|` is below the epsilon
//! of one. Each of those is measured here rather than asserted.
//!
//! The second is the Dekker residual guard. The Best-tier correction in all three functions
//! forms `fma(n, l, -(n * l))`, which is `inf - inf` at the domain edge (`x = -1` here,
//! `x = 0` for `powf_m1`) - so the corrected result was NaN at Precision while Performance,
//! which skips the correction, returned the correct finite limit. The tier a caller picks
//! must not change the answer at the edge, so every edge case below runs at both.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]

mod harness;

use thermite::math::policy::policies::{Performance, Precision};
use thermite::math::{TranscendentalMath, TranscendentalMathWithPolicy};
use thermite::prelude::*;
use thermite::simd::Simd;

macro_rules! ctx {
    () => {
        #[allow(dead_code)]
        type D = Vector<<S as Simd>::f64x4>;
        #[allow(dead_code)]
        type F = Vector<<S as Simd>::f32x8>;

        #[allow(dead_code)]
        #[track_caller]
        fn close(name: &str, got: f64, want: f64, tol: f64) {
            let rel = if want == 0.0 {
                got.abs()
            } else {
                ((got - want) / want).abs()
            };
            assert!(rel <= tol, "{name}: got {got:?}, want {want:?} (rel {rel:e})");
        }

        #[allow(dead_code)]
        fn compound_m1(x: f64, n: f64) -> f64 {
            D::splat(x).compound_m1(D::splat(n)).extract::<0>()
        }

/// `(x, n, (1+x)^n - 1, tolerance in ulps of one)` from mpmath at 60 digits. Computed as
/// `power(1+x, n) - 1` in extended precision, so the reference does not share the kernel's
/// identity.
///
/// The last row's 256 is not slack, it is the measured state of the shared exponent: any
/// absolute error in `n * ln(1 + x)` is relative error in the result, and at `n = 1e4` the
/// single-width error of `ln_1p` itself is multiplied by ten thousand. `compound` carries
/// the same figure (the precision audit records 247 ulp there, and `thermite-interval`
/// budgets 1024 for it); shrinking it needs the double-double log core, not a change here.
#[rustfmt::skip]
#[allow(dead_code)]
const COMPOUND_M1: [(f64, f64, f64, f64); 12] = [
    (-0.9,      3.0,     -0.999,                  16.0),
    (-0.5,      2.5,     -0.8232233047033631,     16.0),
    (-0.25,    -4.0,      2.1604938271604937,     16.0),
    (1e-20,     0.5,      5e-21,                  16.0),
    (1e-12,     2.0,      2.000000000001e-12,     16.0),
    (0.5,       1e-18,    4.054651081081644e-19,  16.0),
    (0.5,       3.0,      2.375,                  16.0),
    (2.0,      -1.5,     -0.8075499102701248,     16.0),
    (100000.0,  0.25,    16.782838557207768,      16.0),
    (-1e-16,    7.0,     -6.999999999999998e-16,  16.0),
    (3.0,       0.0,      0.0,                    16.0),
    (0.05,      10000.0,  7.81611065842881e211,  256.0),
];
    };
}

for_each_backend_concrete! {

fn matches_the_reference() {
    ctx!();
    for &(x, n, want, ulps) in COMPOUND_M1.iter() {
        close(
            &format!("compound_m1({x}, {n})"),
            compound_m1(x, n),
            want,
            ulps * f64::EPSILON,
        );
    }
}

fn the_precision_tier_agrees_with_the_reference_too() {
    ctx!();
    // The Dekker correction only runs at Best and above, so the table above exercises one
    // arm of the function. This runs the other.
    for &(x, n, want, ulps) in COMPOUND_M1.iter() {
        let got = D::splat(x).compound_m1_p::<Precision>(D::splat(n)).extract::<0>();
        close(&format!("prec compound_m1({x}, {n})"), got, want, ulps * f64::EPSILON);
    }
}

fn small_x_survives_where_the_powf_m1_spelling_cannot() {
    ctx!();
    // Forming 1 + x rounds x away entirely below the epsilon of one, so the alternative
    // spelling returns a flat zero. Measured, not claimed.
    for &(x, n) in &[(1e-20_f64, 0.5_f64), (1e-18, 3.0), (-1e-19, 2.0)] {
        let want = n * x; // (1+x)^n - 1 = n x + O(x^2), and x^2 is far below the ulp here
        close(&format!("compound_m1({x}, {n})"), compound_m1(x, n), want, 1e-14);

        assert_eq!(
            (1.0f64 + x).powf(n) - 1.0,
            0.0,
            "precondition: powf(1 + {x}, {n}) - 1 collapses"
        );
    }
}

fn a_near_zero_result_survives_where_the_compound_spelling_cannot() {
    ctx!();
    // The other end: x is ordinary but n is tiny, so (1+x)^n is a hair above one and
    // subtracting one afterwards cancels everything.
    for &(x, n) in &[(0.5_f64, 1e-18_f64), (3.0, 1e-20), (-0.5, 1e-17)] {
        let want = n * (1.0 + x).ln();
        close(&format!("compound_m1({x}, {n})"), compound_m1(x, n), want, 1e-14);

        assert_eq!(
            (1.0f64 + x).powf(n) - 1.0,
            0.0,
            "precondition: pow(1 + {x}, {n}) - 1 collapses"
        );
    }
}

fn the_domain_edge_is_the_limit_at_every_tier() {
    ctx!();
    // x = -1 is the edge: (1+x)^n = 0^n, so the answer is -1 for n > 0 and +inf for n < 0.
    // The Best-tier residual is inf - inf there, so without a guard the high tier returns
    // NaN where the cheap one is right, and the answer depends on the policy.
    for &n in &[3.0_f64, 0.5, 1.0, 1e-3] {
        let (perf, prec) = (
            D::splat(-1.0).compound_m1_p::<Performance>(D::splat(n)).extract::<0>(),
            D::splat(-1.0).compound_m1_p::<Precision>(D::splat(n)).extract::<0>(),
        );
        assert_eq!(perf, -1.0, "compound_m1(-1, {n}) at Performance");
        assert_eq!(prec, -1.0, "compound_m1(-1, {n}) at Precision");
    }

    for &n in &[-1.5_f64, -2.0, -0.25] {
        assert_eq!(
            D::splat(-1.0).compound_m1_p::<Performance>(D::splat(n)).extract::<0>(),
            f64::INFINITY
        );
        assert_eq!(
            D::splat(-1.0).compound_m1_p::<Precision>(D::splat(n)).extract::<0>(),
            f64::INFINITY
        );
    }

    // Below the edge there is no real value.
    assert!(compound_m1(-1.5, 2.0).is_nan());
    assert!(compound_m1(f64::NAN, 2.0).is_nan());
    assert!(compound_m1(1.0, f64::NAN).is_nan());
}

fn the_siblings_edges_are_the_limit_at_every_tier() {
    ctx!();
    // The same residual guard in `compound` and `powf_m1`. `compound(-1, n) = 0^n` and
    // `powf_m1(0, e) = 0^e - 1`, both finite limits that Precision has to reach too.
    for &n in &[2.0_f64, 0.5, 3.0] {
        assert_eq!(
            D::splat(-1.0).compound_p::<Performance>(D::splat(n)).extract::<0>(),
            0.0
        );
        assert_eq!(D::splat(-1.0).compound_p::<Precision>(D::splat(n)).extract::<0>(), 0.0);

        assert_eq!(D::splat(0.0).powf_m1_p::<Performance>(D::splat(n)).extract::<0>(), -1.0);
        assert_eq!(D::splat(0.0).powf_m1_p::<Precision>(D::splat(n)).extract::<0>(), -1.0);
    }

    for &n in &[-1.5_f64, -2.0] {
        assert_eq!(
            D::splat(-1.0).compound_p::<Precision>(D::splat(n)).extract::<0>(),
            f64::INFINITY
        );
        assert_eq!(
            D::splat(0.0).powf_m1_p::<Precision>(D::splat(n)).extract::<0>(),
            f64::INFINITY
        );
    }
}

fn it_agrees_with_compound_away_from_the_cancelling_regions() {
    ctx!();
    // Where neither spelling is in trouble the two must not disagree, which catches a sign
    // or an off-by-one in the -1.
    for &(x, n) in &[(0.5_f64, 3.0_f64), (2.0, -1.5), (-0.5, 2.5), (100.0, 0.25)] {
        let a = compound_m1(x, n);
        let b = D::splat(x).compound(D::splat(n)).extract::<0>() - 1.0;
        close(&format!("compound_m1 vs compound - 1 at ({x}, {n})"), a, b, 1e-14);
    }
}

fn f32_tracks_the_reference() {
    ctx!();
    for &(x, n, want, _) in COMPOUND_M1.iter() {
        if want.abs() > 1e30 || (x != 0.0 && (x as f32) == 0.0) {
            continue;
        }
        let got = F::splat(x as f32).compound_m1(F::splat(n as f32)).extract::<0>() as f64;
        close(
            &format!("f32 compound_m1({x}, {n})"),
            got,
            want,
            64.0 * f32::EPSILON as f64,
        );
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn lanes_stay_independent_on_a_wide_backend() {
    ctx!();
    use thermite::simd::Simd;
    type W = Vector<<S as Simd>::f64x4>;

    // One lane at the domain edge, one in each cancelling region, one ordinary.
    let xs = [-1.0, 1e-20, 0.5, 2.0];
    let ns = [3.0, 0.5, 1e-18, -1.5];

    let got = W::from_slice(&xs).compound_m1(W::from_slice(&ns)).into_array();
    let prec = W::from_slice(&xs)
        .compound_m1_p::<Precision>(W::from_slice(&ns))
        .into_array();

    for lane in 0..4 {
        assert_eq!(got.as_slice()[lane], compound_m1(xs[lane], ns[lane]), "lane {lane}");
        close(
            &format!("wide precision lane {lane}"),
            prec.as_slice()[lane],
            compound_m1(xs[lane], ns[lane]),
            16.0 * f64::EPSILON,
        );
    }
}

}
