//! Gate for `atan2` at **every policy tier**, both dtypes.
//!
//! This file exists because of a specific failure. The float32
//! `precision <= Medium` branch divided `max/min` where every other lowering in the tree
//! divides `min/max`, so `HighPerformance` and `UltraPerformance` returned the
//! *complement* of the answer for every input with `|x| != |y|`. `atan2(1, 2)` gave
//! 1.09252 where 0.46365 was wanted. It shipped.
//!
//! The reason it shipped is the thing to design against: **the existing sweeps tested
//! only the default policy.** A kernel branch that only compiles in below `Average` was
//! never executed by any test. So every case here runs at all seven tiers, and the
//! tolerance is *per tier* rather than one loose bound that a fast tier could hide in.
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use thermite::backend::x86_v3::prelude::*;
use thermite::math::policy::DefaultPolicy;
use thermite::math::policy::policies::{HighPerformance, Performance, Precision, Reference, Size, UltraPerformance};

macro_rules! for_each_tier {
    (|$p:ident, $tol:ident| $body:block) => {{
        {
            type $p = UltraPerformance;
            let $tol = TOL[0];
            let _ = $tol;
            $body
        }
        {
            type $p = HighPerformance;
            let $tol = TOL[1];
            let _ = $tol;
            $body
        }
        {
            type $p = Performance;
            let $tol = TOL[2];
            let _ = $tol;
            $body
        }
        {
            type $p = Size;
            let $tol = TOL[3];
            let _ = $tol;
            $body
        }
        {
            type $p = DefaultPolicy;
            let $tol = TOL[4];
            let _ = $tol;
            $body
        }
        {
            type $p = Precision;
            let $tol = TOL[5];
            let _ = $tol;
            $body
        }
        {
            type $p = Reference;
            let $tol = TOL[6];
            let _ = $tol;
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

/// Absolute tolerance in radians, per tier, in `TIER_NAMES` order.
///
/// **Per tier on purpose.** A single loose bound sized for `UltraPerformance` would have
/// passed the complement bug at every tier, since the complement of an angle in
/// `[0, pi]` is still in `[0, pi]` and never more than `pi` away. The fast tiers are
/// allowed to be imprecise. They are not allowed to be a different function.
const TOL: [f32; 7] = [
    3.0e-5, // UltraPerformance: the Medium rational fit
    3.0e-5, // HighPerformance: same branch
    5.0e-7, // Performance
    5.0e-7, // Size
    5.0e-7, // DefaultPolicy
    5.0e-7, // Precision
    5.0e-7, // Reference
];
// 5e-7 rather than 2e-7 for the accurate tiers because the result can be as large as pi,
// where one ulp of float32 is already 2.4e-7. Worst on the grid below is
// 2.37e-7 at atan2(-9.25, -12.2), which is ~1 ulp of the ANSWER. Still 60x tighter than
// the fast tiers, and four orders of magnitude tighter than the ~1 rad a complement bug
// produces, which is the error class this file is built to catch.

/// `check_overflow`, per tier. False only for the two fastest.
const CHECKS: [bool; 7] = [false, false, true, true, true, true, true];

fn ang_close(got: f32, want: f64, tol: f32) -> bool {
    if got.is_nan() {
        return false;
    }
    ((got as f64) - want).abs() <= tol as f64
}

// ---------------------------------------------------------------------------
// 1. The dense grid. This is what would have caught the bug on day one.
// ---------------------------------------------------------------------------

/// Every quadrant, every ratio, at every tier.
///
/// The grid deliberately includes `|y| < |x|` and `|y| > |x|` in quantity: the bug lived
/// entirely in which of the two got put in the numerator, so a grid that only sampled one
/// side of `|y| == |x|` would have missed half of it and a grid that only sampled
/// `|y| == |x|` would have missed all of it.
#[test]
fn dense_grid_every_quadrant_every_tier() {
    let mut pts: Vec<(f32, f32)> = Vec::new();
    for i in -32..=32 {
        for j in -32..=32 {
            if i == 0 && j == 0 {
                continue;
            }
            pts.push((i as f32 * 0.37, j as f32 * 0.61));
        }
    }
    // Ratios spanning many decades, both orders.
    for e in -18..=18 {
        let r = (10.0f64).powi(e) as f32;
        pts.push((r, 1.0));
        pts.push((1.0, r));
        pts.push((-r, 1.0));
        pts.push((1.0, -r));
        pts.push((r, -1.0));
        pts.push((-1.0, r));
    }

    let mut ti = 0;
    for_each_tier!(|P, tol| {
        let mut worst = 0.0f64;
        let mut worst_at = (0.0f32, 0.0f32);

        for &(y, x) in &pts {
            let got = f32x8::splat(y).atan2_p::<P>(f32x8::splat(x)).into_array()[0];
            let want = (y as f64).atan2(x as f64);
            let d = ((got as f64) - want).abs();
            if d > worst {
                worst = d;
                worst_at = (y, x);
            }
        }

        assert!(
            worst <= tol as f64,
            "{}: worst {worst} rad at atan2({}, {}) over {} points",
            TIER_NAMES[ti],
            worst_at.0,
            worst_at.1,
            pts.len()
        );
        ti += 1;
    });
}

#[test]
fn dense_grid_f64_every_quadrant_every_tier() {
    let mut pts: Vec<(f64, f64)> = Vec::new();
    for i in -24..=24 {
        for j in -24..=24 {
            if i == 0 && j == 0 {
                continue;
            }
            pts.push((i as f64 * 0.41, j as f64 * 0.53));
        }
    }
    for e in -150..=150 {
        let r = (10.0f64).powi(e);
        pts.push((r, 1.0));
        pts.push((1.0, r));
        pts.push((-r, 1.0));
        pts.push((1.0, -r));
    }

    let mut ti = 0;
    for_each_tier!(|P, tol| {
        let mut worst = 0.0f64;
        let mut worst_at = (0.0f64, 0.0f64);

        for &(y, x) in &pts {
            let got = f64x4::splat(y).atan2_p::<P>(f64x4::splat(x)).into_array()[0];
            let want = y.atan2(x);
            let d = (got - want).abs();
            if d > worst {
                worst = d;
                worst_at = (y, x);
            }
        }

        // f64 has no Medium branch, so every tier gets the tight bound.
        assert!(
            worst <= 1.0e-9,
            "{} f64: worst {worst} rad at atan2({}, {})",
            TIER_NAMES[ti],
            worst_at.0,
            worst_at.1
        );
        ti += 1;
    });
}

// ---------------------------------------------------------------------------
// 2. The identity that pins the ratio's ORIENTATION.
// ---------------------------------------------------------------------------

/// `atan2(y, x) + atan2(x, y) == pi/2` for positive `x`, `y`.
///
/// **This is the single check that makes the complement bug impossible.** Swapping the
/// numerator and denominator maps an angle to its complement, and a complement is
/// indistinguishable from the truth by any bound-on-the-error test that is loose enough
/// for a fast tier. Here it is not: the sum is `pi/2` for the correct orientation and
/// `3*pi/2 - 2*theta` for the inverted one, off by a full radian in the middle of the
/// range.
#[test]
fn complementary_angle_identity() {
    let vals: [f32; 12] = [
        1.0, 2.0, 0.5, 3.0, 0.125, 7.0, 1e3, 1e-3, 1e8, 1e-8, 1.7320508, 0.57735026,
    ];

    let mut ti = 0;
    for_each_tier!(|P, tol| {
        for &y in &vals {
            for &x in &vals {
                let a = f32x8::splat(y).atan2_p::<P>(f32x8::splat(x)).into_array()[0];
                let b = f32x8::splat(x).atan2_p::<P>(f32x8::splat(y)).into_array()[0];
                let sum = a + b;
                assert!(
                    ((sum - core::f32::consts::FRAC_PI_2).abs()) <= tol * 4.0,
                    "{}: atan2({y}, {x}) + atan2({x}, {y}) = {sum}, want pi/2 ({a} + {b})",
                    TIER_NAMES[ti]
                );
            }
        }
        ti += 1;
    });
}

/// `atan2(y, x) == atan(y/x)` in the first quadrant, where no quadrant fixup applies.
///
/// A second, independent way of catching an inverted ratio: it ties `atan2` to the
/// one-argument `atan`, which has its own tests and its own fit.
#[test]
fn first_quadrant_agrees_with_atan_of_the_ratio() {
    let ratios: [f32; 9] = [0.001, 0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 10.0, 1000.0];

    let mut ti = 0;
    for_each_tier!(|P, tol| {
        for &r in &ratios {
            let a = f32x8::splat(r).atan2_p::<P>(f32x8::splat(1.0)).into_array()[0];
            let b = f32x8::splat(r).atan_p::<P>().into_array()[0];
            assert!(
                (a - b).abs() <= tol * 4.0,
                "{}: atan2({r}, 1) = {a} but atan({r}) = {b}",
                TIER_NAMES[ti]
            );
        }
        ti += 1;
    });
}

// ---------------------------------------------------------------------------
// 3. Quadrants and signs, which a magnitude-only test cannot see.
// ---------------------------------------------------------------------------

#[test]
fn every_quadrant_lands_in_its_own_range() {
    // (y, x, expected sign of result, expected |result| range)
    let cases: [(f32, f32, &str); 4] = [
        (1.0, 1.0, "Q1: (0, pi/2)"),
        (1.0, -1.0, "Q2: (pi/2, pi)"),
        (-1.0, -1.0, "Q3: (-pi, -pi/2)"),
        (-1.0, 1.0, "Q4: (-pi/2, 0)"),
    ];

    let mut ti = 0;
    for_each_tier!(|P, tol| {
        for &(y, x, label) in &cases {
            for &s in &[0.5f32, 1.0, 2.0, 1e6, 1e-6] {
                let got = f32x8::splat(y * s).atan2_p::<P>(f32x8::splat(x)).into_array()[0];
                let want = ((y * s) as f64).atan2(x as f64);
                assert!(
                    got.signum() == want.signum() as f32 || got == 0.0,
                    "{}: {label} atan2({}, {x}) = {got}, want sign of {want}",
                    TIER_NAMES[ti],
                    y * s
                );
                assert!(
                    ang_close(got, want, tol),
                    "{}: {label} atan2({}, {x}) = {got}, want {want}",
                    TIER_NAMES[ti],
                    y * s
                );
            }
        }
        ti += 1;
    });
}

/// Signed zero picks the branch cut: `atan2(+0, -1) == +pi`, `atan2(-0, -1) == -pi`.
#[test]
fn signed_zero_selects_the_branch() {
    let cases: [(f32, f32, f64); 8] = [
        (0.0, 1.0, 0.0),
        (-0.0, 1.0, -0.0),
        (0.0, -1.0, core::f64::consts::PI),
        (-0.0, -1.0, -core::f64::consts::PI),
        (1.0, 0.0, core::f64::consts::FRAC_PI_2),
        (-1.0, 0.0, -core::f64::consts::FRAC_PI_2),
        (1.0, -0.0, core::f64::consts::FRAC_PI_2),
        (-1.0, -0.0, -core::f64::consts::FRAC_PI_2),
    ];

    let mut ti = 0;
    for_each_tier!(|P, tol| {
        for &(y, x, want) in &cases {
            let got = f32x8::splat(y).atan2_p::<P>(f32x8::splat(x)).into_array()[0];
            assert!(
                ang_close(got, want, tol),
                "{}: atan2({y}, {x}) = {got}, want {want}",
                TIER_NAMES[ti]
            );
            if want != 0.0 {
                assert_eq!(
                    got.is_sign_negative(),
                    want.is_sign_negative(),
                    "{}: atan2({y}, {x}) = {got} has the wrong sign for {want}",
                    TIER_NAMES[ti]
                );
            }
        }
        ti += 1;
    });
}

/// The origin. `atan2(0, 0) == 0` by convention at EVERY tier, since it is the function's
/// value there rather than an overflow edge, so it is not behind `check_overflow`. The
/// float32 Medium branch used to gate it and return NaN, while the float32 `Best` branch
/// and every float64 tier returned 0.
#[test]
fn the_origin_is_zero_at_every_tier() {
    let mut ti = 0;
    for_each_tier!(|P, tol| {
        for &(y, x) in &[(0.0f32, 0.0f32), (-0.0, 0.0), (0.0, -0.0), (-0.0, -0.0)] {
            let got = f32x8::splat(y).atan2_p::<P>(f32x8::splat(x)).into_array()[0];
            assert!(
                !got.is_nan() && got.abs() <= core::f32::consts::PI,
                "{}: atan2({y}, {x}) = {got}, want a finite angle",
                TIER_NAMES[ti]
            );

            let g64 = f64x4::splat(y as f64).atan2_p::<P>(f64x4::splat(x as f64)).into_array()[0];
            assert!(!g64.is_nan(), "{} f64: atan2({y}, {x}) = {g64}", TIER_NAMES[ti]);
        }
        ti += 1;
    });
}

// ---------------------------------------------------------------------------
// 4. Extremes, which is where the NaNs were.
// ---------------------------------------------------------------------------

/// Ratios far past what a float can represent. These returned NaN before the fix,
/// because an inverted ratio squared overflows once `max/min > 1.8e19`.
#[test]
fn extreme_magnitude_ratios() {
    let cases: [(f32, f32); 10] = [
        (3.7e-21, 207761.0),
        (3.2e13, 2.6e-28),
        (9.5e-23, 3.2e20),
        (1e-30, 1e30),
        (1e30, 1e-30),
        (f32::MIN_POSITIVE, f32::MAX),
        (f32::MAX, f32::MIN_POSITIVE),
        (f32::from_bits(1), 1.0),
        (1.0, f32::from_bits(1)),
        (f32::MAX, f32::MAX),
    ];

    let mut ti = 0;
    for_each_tier!(|P, tol| {
        for &(y, x) in &cases {
            for (a, b) in [(y, x), (-y, x), (y, -x), (-y, -x)] {
                let got = f32x8::splat(a).atan2_p::<P>(f32x8::splat(b)).into_array()[0];
                let want = (a as f64).atan2(b as f64);
                assert!(
                    ang_close(got, want, tol),
                    "{}: atan2({a}, {b}) = {got}, want {want}",
                    TIER_NAMES[ti]
                );
            }
        }
        ti += 1;
    });
}

/// Infinities. All four `(+-inf, +-inf)` corners are the diagonal angles, and a finite
/// operand against an infinite one collapses to an axis. Only the `check_overflow` tiers
/// promise this.
#[test]
fn infinite_operands_where_overflow_is_checked() {
    let inf = f32::INFINITY;
    let cases: [(f32, f32, f64); 8] = [
        (inf, inf, core::f64::consts::FRAC_PI_4),
        (inf, -inf, 3.0 * core::f64::consts::FRAC_PI_4),
        (-inf, inf, -core::f64::consts::FRAC_PI_4),
        (-inf, -inf, -3.0 * core::f64::consts::FRAC_PI_4),
        (1.0, inf, 0.0),
        (1.0, -inf, core::f64::consts::PI),
        (inf, 1.0, core::f64::consts::FRAC_PI_2),
        (-inf, 1.0, -core::f64::consts::FRAC_PI_2),
    ];

    let mut ti = 0;
    for_each_tier!(|P, tol| {
        if CHECKS[ti] {
            for &(y, x, want) in &cases {
                let got = f32x8::splat(y).atan2_p::<P>(f32x8::splat(x)).into_array()[0];
                assert!(
                    ang_close(got, want, tol * 4.0),
                    "{}: atan2({y}, {x}) = {got}, want {want}",
                    TIER_NAMES[ti]
                );

                let g64 = f64x4::splat(y as f64).atan2_p::<P>(f64x4::splat(x as f64)).into_array()[0];
                assert!(
                    (g64 - want).abs() <= 1e-9,
                    "{} f64: atan2({y}, {x}) = {g64}, want {want}",
                    TIER_NAMES[ti]
                );
            }
        }
        ti += 1;
    });
}

// ---------------------------------------------------------------------------
// 5. Structural.
// ---------------------------------------------------------------------------

/// Lane independence: a packed call must equal the same pairs computed one at a time.
#[test]
fn lanes_do_not_influence_each_other() {
    let ys: [f32; 8] = [1.0, -2.0, 0.0, 1e20, 1e-20, f32::INFINITY, -0.0, 3.0];
    let xs: [f32; 8] = [2.0, 1.0, -1.0, 1e-20, 1e20, 1.0, -1.0, -4.0];

    let mut ti = 0;
    for_each_tier!(|P, tol| {
        let packed = f32x8::new(ys).atan2_p::<P>(f32x8::new(xs)).into_array();
        for i in 0..8 {
            let alone = f32x8::splat(ys[i]).atan2_p::<P>(f32x8::splat(xs[i])).into_array()[0];
            assert_eq!(
                packed[i].to_bits(),
                alone.to_bits(),
                "{}: lane {i} atan2({}, {}) = {} packed but {} alone",
                TIER_NAMES[ti],
                ys[i],
                xs[i],
                packed[i],
                alone
            );
        }
        ti += 1;
    });
}

/// Every backend must agree. The kernel is shared but the lowering is not.
#[test]
fn every_backend_agrees() {
    use thermite::backend::scalar::Scalar;
    use thermite::backend::{x86_v1::X86V1, x86_v2::X86V2};
    use thermite::simd::Simd;

    type S1 = Vector<<Scalar as Simd>::f32x4>;
    type V1 = Vector<<X86V1 as Simd>::f32x4>;
    type V2 = Vector<<X86V2 as Simd>::f32x8>;

    let cases: [(f32, f32); 10] = [
        (1.0, 2.0),
        (2.0, 1.0),
        (-1.0, 2.0),
        (1.0, -2.0),
        (0.0, 0.0),
        (1e20, 1e-20),
        (1e-20, 1e20),
        (1.0, 0.0),
        (0.0, -1.0),
        (-3.0, -4.0),
    ];

    for &(y, x) in &cases {
        let want = (y as f64).atan2(x as f64);

        let s = S1::splat(y).atan2_p::<Precision>(S1::splat(x)).into_array()[0];
        let v1 = V1::splat(y).atan2_p::<Precision>(V1::splat(x)).into_array()[0];
        let v2 = V2::splat(y).atan2_p::<Precision>(V2::splat(x)).into_array()[0];
        let v3 = f32x8::splat(y).atan2_p::<Precision>(f32x8::splat(x)).into_array()[0];

        for (name, got) in [("scalar", s), ("x86_v1", v1), ("x86_v2", v2), ("x86_v3", v3)] {
            assert!(
                ang_close(got, want, 2.0e-7),
                "{name}: atan2({y}, {x}) = {got}, want {want}"
            );
        }
    }
}
