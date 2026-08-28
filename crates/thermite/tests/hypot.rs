//! Exhaustive-ish gate for `hypot`, `hypot_n` and `inv_hypot_n`.
//!
//! `hypot` is a *range* function: its whole reason to exist is that
//! `sqrt(x*x + y*y)` overflows and underflows long before the answer does. So the
//! interesting inputs are the ones a normal sweep never generates (operands 200
//! binades apart, subnormals, values one ulp from the format's limits), and the
//! interesting property is not "how many ulps" but "does it survive at all".
//!
//! Everything here runs at **every policy tier**, because the kernel
//! is policy-invariant by design. A tier appearing in a failure message is the point:
//! it means the invariance broke.
#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use thermite::backend::x86_v3::prelude::*;
use thermite::math::SpatialMathWithPolicy;
use thermite::math::policy::DefaultPolicy;
use thermite::math::policy::policies::{HighPerformance, Performance, Precision, Reference, Size, UltraPerformance};

/// Every tier, run through one closure. `$p` is the `Policy` type parameter, and `$ti` is
/// the tier's index into `TIER_NAMES`, owned by the macro so call sites don't carry a
/// manual counter whose last increment trips `unused_assignments`.
macro_rules! for_each_tier {
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

fn ulps_f32(got: f32, want: f32) -> f64 {
    if got.is_nan() || want.is_nan() {
        return if got.is_nan() == want.is_nan() {
            0.0
        } else {
            f64::INFINITY
        };
    }
    if !got.is_finite() || !want.is_finite() {
        return if got == want { 0.0 } else { f64::INFINITY };
    }
    if want == 0.0 {
        return if got == 0.0 { 0.0 } else { f64::INFINITY };
    }
    (((got as f64) - (want as f64)) / ((want.abs() as f64) * f64::from(f32::EPSILON))).abs()
}

fn ulps_f64(got: f64, want: f64) -> f64 {
    if got.is_nan() || want.is_nan() {
        return if got.is_nan() == want.is_nan() {
            0.0
        } else {
            f64::INFINITY
        };
    }
    if !got.is_finite() || !want.is_finite() {
        return if got == want { 0.0 } else { f64::INFINITY };
    }
    if want == 0.0 {
        return if got == 0.0 { 0.0 } else { f64::INFINITY };
    }
    ((got - want) / (want.abs() * f64::EPSILON)).abs()
}

/// Tolerance. `inverse_sqrt` is an approximate instruction plus a Newton step below
/// `Best`, so the inverse form gets a relative budget rather than a ULP one.
const ULP_TOL: f64 = 2.0;
const INV_REL_TOL: f32 = 1e-2;

// ---------------------------------------------------------------------------
// 1. The full exponent cross-product, both dtypes, every tier.
// ---------------------------------------------------------------------------

#[test]
fn f32_every_binade_pair_at_every_tier() {
    // -149 is the smallest subnormal, 127 the largest binade. Step 3 with the
    // near-equal band filled in: far-apart pairs answer trivially (the result IS the
    // larger operand), so the small offsets are where the arithmetic happens.
    let mut exps: Vec<i32> = (-149..=127).step_by(3).collect();
    exps.extend(-6..=6);
    exps.sort_unstable();
    exps.dedup();

    let mants: [f32; 3] = [1.0, 1.4142135, 1.9999999];

    for_each_tier!(|P, ti| {
        let mut worst = 0.0f64;
        let mut worst_at = (0.0f32, 0.0f32);
        let mut n = 0usize;

        for &a in &exps {
            for &b in &exps {
                for &mx in &mants {
                    let xv = (2.0f64).powi(a) as f32 * mx;
                    let yv = (2.0f64).powi(b) as f32 * mants[2 - mants.iter().position(|m| *m == mx).unwrap()];
                    let got = f32x8::splat(xv).hypot_p::<P>(f32x8::splat(yv)).into_array();
                    let want = libm::hypotf(xv, yv);

                    for &g in got.iter() {
                        let u = ulps_f32(g, want);
                        if u > worst {
                            worst = u;
                            worst_at = (xv, yv);
                        }
                    }
                    n += 1;
                }
            }
        }

        assert!(
            worst <= ULP_TOL,
            "{} f32: worst {worst} ulp at hypot({}, {}) over {n} pairs",
            TIER_NAMES[ti],
            worst_at.0,
            worst_at.1
        );
    });
}

#[test]
fn f64_every_binade_pair_at_every_tier() {
    let mut exps: Vec<i32> = (-1074..=1023).step_by(29).collect();
    exps.extend(-6..=6);
    exps.sort_unstable();
    exps.dedup();

    let mants: [f64; 2] = [1.0, 1.4142135623730951];

    for_each_tier!(|P, ti| {
        let mut worst = 0.0f64;
        let mut worst_at = (0.0f64, 0.0f64);

        for &a in &exps {
            for &b in &exps {
                for &mx in &mants {
                    let xv = (2.0f64).powi(a) * mx;
                    let yv = (2.0f64).powi(b) * 1.7320508075688772;
                    let got = f64x4::splat(xv).hypot_p::<P>(f64x4::splat(yv)).into_array();
                    let want = libm::hypot(xv, yv);

                    for &g in got.iter() {
                        let u = ulps_f64(g, want);
                        if u > worst {
                            worst = u;
                            worst_at = (xv, yv);
                        }
                    }
                }
            }
        }

        assert!(
            worst <= ULP_TOL,
            "{} f64: worst {worst} ulp at hypot({}, {})",
            TIER_NAMES[ti],
            worst_at.0,
            worst_at.1
        );
    });
}

// ---------------------------------------------------------------------------
// 2. The specific values that used to break.
// ---------------------------------------------------------------------------

/// Every one of these returned a wrong answer somewhere before the rewrite.
#[test]
fn the_regressions_that_motivated_the_rewrite() {
    // (x, y, what was wrong before)
    let cases: [(f32, f32, &str); 8] = [
        (3.0e38, 4.0e37, "overflowed to inf at UltraPerformance"),
        (1.0e30, 1.0e30, "overflowed to inf at UltraPerformance"),
        (9.5e-23, 3.2e20, "wide spread overflowed at UltraPerformance"),
        (1.0e-40, 1.0e-40, "underflowed to 0.0 at UltraPerformance"),
        (7.0e-39, 1.0e-45, "underflowed to 0.0 at UltraPerformance"),
        (f32::from_bits(1), 0.0, "smallest subnormal"),
        (0.0, 0.0, "zero guard"),
        (f32::MAX, f32::MIN_POSITIVE, "widest finite spread"),
    ];

    for_each_tier!(|P, ti| {
        for &(x, y, why) in &cases {
            let want = libm::hypotf(x, y);
            for (a, b) in [(x, y), (y, x)] {
                let got = f32x8::splat(a).hypot_p::<P>(f32x8::splat(b)).into_array()[0];
                assert!(
                    ulps_f32(got, want) <= ULP_TOL,
                    "{}: hypot({a}, {b}) = {got}, want {want} [{why}]",
                    TIER_NAMES[ti]
                );
            }
        }
    });
}

// ---------------------------------------------------------------------------
// 3. Non-finite behaviour, which is where the policy split actually lives.
// ---------------------------------------------------------------------------

/// NaN must survive in EITHER operand position.
///
/// The old kernel lost it asymmetrically: `maxps`/`minps` return their *second* operand
/// when either input is NaN, so `hypot(NaN, 1.0)` came back **1.4142135** and
/// `hypot(NaN, 0.0)` came back **0.0**, while `hypot(1.0, NaN)` was correctly NaN.
#[test]
fn nan_propagates_from_either_operand_at_every_tier() {
    let nan = f32::NAN;
    let others: [f32; 6] = [0.0, 1.0, -1.0, f32::MIN_POSITIVE, f32::MAX, nan];

    for_each_tier!(|P, ti| {
        for &o in &others {
            for (a, b) in [(nan, o), (o, nan)] {
                let got = f32x8::splat(a).hypot_p::<P>(f32x8::splat(b)).into_array()[0];
                assert!(got.is_nan(), "{}: hypot({a}, {b}) = {got}, want NaN", TIER_NAMES[ti]);
            }
        }
    });
}

/// `hypot(+-inf, y) == +inf` for ANY finite or NaN `y` (C99 F.10.4.3), on the tiers
/// that check overflow. The tiers that do not are asserted to at least be finite-or-inf
/// rather than silently something else.
#[test]
fn infinity_dominates_where_overflow_is_checked() {
    let inf = f32::INFINITY;
    let others: [f32; 5] = [0.0, 1.0, -3.5, f32::MAX, f32::NAN];

    // check_overflow is false for UltraPerformance and HighPerformance only.
    let checks: [bool; 7] = [false, false, true, true, true, true, true];

    for_each_tier!(|P, ti| {
        for &o in &others {
            for (a, b) in [(inf, o), (o, inf), (-inf, o), (o, -inf)] {
                let got = f32x8::splat(a).hypot_p::<P>(f32x8::splat(b)).into_array()[0];
                if checks[ti] {
                    assert!(
                        got.is_infinite() && got.is_sign_positive(),
                        "{}: hypot({a}, {b}) = {got}, want +inf",
                        TIER_NAMES[ti]
                    );
                } else if !o.is_nan() {
                    // Without the check, an infinite operand still gives infinity by
                    // arithmetic, and only the NaN pairing is allowed to differ.
                    assert!(got.is_infinite(), "{}: hypot({a}, {b}) = {got}", TIER_NAMES[ti]);
                }
            }
        }
    });
}

/// Sign is dropped: `hypot` is a magnitude, so every result is non-negative.
#[test]
fn the_result_is_never_negative() {
    let vals: [f32; 9] = [-0.0, 0.0, -1.0, 1.0, -1e30, 1e30, -1e-40, 1e-40, -f32::MAX];

    for_each_tier!(|P, ti| {
        for &a in &vals {
            for &b in &vals {
                let got = f32x8::splat(a).hypot_p::<P>(f32x8::splat(b)).into_array()[0];
                assert!(
                    !got.is_sign_negative() || got.is_nan(),
                    "{}: hypot({a}, {b}) = {got}, must be non-negative",
                    TIER_NAMES[ti]
                );
            }
        }
    });
}

// ---------------------------------------------------------------------------
// 4. Algebraic identities, which catch classes of error a value table cannot.
// ---------------------------------------------------------------------------

/// `hypot` is symmetric in its arguments, exactly, not to a tolerance.
///
/// The old `min`/`max` form was NOT: with a NaN operand the answer depended on which
/// side it arrived on.
#[test]
fn hypot_is_exactly_symmetric() {
    let mut vals: Vec<f32> = Vec::new();
    for e in (-149..=127).step_by(7) {
        vals.push((2.0f64).powi(e) as f32);
        vals.push(-((2.0f64).powi(e) as f32) * 1.618034);
    }
    vals.extend([0.0, -0.0, f32::NAN, f32::INFINITY, -f32::INFINITY, f32::MAX]);

    for_each_tier!(|P, ti| {
        for &a in &vals {
            for &b in &vals {
                let ab = f32x8::splat(a).hypot_p::<P>(f32x8::splat(b)).into_array()[0];
                let ba = f32x8::splat(b).hypot_p::<P>(f32x8::splat(a)).into_array()[0];
                assert_eq!(
                    ab.to_bits(),
                    ba.to_bits(),
                    "{}: hypot({a}, {b}) = {ab} but hypot({b}, {a}) = {ba}",
                    TIER_NAMES[ti]
                );
            }
        }
    });
}

/// Scaling both operands by a power of two scales the result by the same factor,
/// exactly. This is the property the rescaling implementation is built on, so if it
/// ever stops holding the implementation is wrong in its core assumption.
#[test]
fn scaling_by_a_power_of_two_is_exact() {
    let pairs: [(f32, f32); 6] = [
        (3.0, 4.0),
        (1.0, 1.0),
        (1.0, 1e-8),
        (1.5, 2.25),
        (1.0, 0.0),
        (7.0, 24.0),
    ];

    for_each_tier!(|P, ti| {
        for &(x, y) in &pairs {
            let base = f32x8::splat(x).hypot_p::<P>(f32x8::splat(y)).into_array()[0];

            for k in -60i32..=60 {
                let f = (2.0f64).powi(k) as f32;
                let got = f32x8::splat(x * f).hypot_p::<P>(f32x8::splat(y * f)).into_array()[0];
                let want = base * f;
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{}: hypot({}, {}) = {got}, but 2^{k} * hypot({x}, {y}) = {want}",
                    TIER_NAMES[ti],
                    x * f,
                    y * f
                );
            }
        }
    });
}

/// `hypot(x, 0) == |x|`, exactly, for every representable magnitude.
#[test]
fn hypot_with_zero_is_the_absolute_value() {
    for_each_tier!(|P, ti| {
        for e in -149..=127 {
            let x = (2.0f64).powi(e) as f32;
            for &s in &[1.0f32, -1.0] {
                for &z in &[0.0f32, -0.0] {
                    let got = f32x8::splat(x * s).hypot_p::<P>(f32x8::splat(z)).into_array()[0];
                    assert_eq!(
                        got.to_bits(),
                        x.to_bits(),
                        "{}: hypot({}, {z}) = {got}",
                        TIER_NAMES[ti],
                        x * s
                    );
                }
            }
        }
    });
}

/// The result is bracketed: `max(|x|,|y|) <= hypot <= max * sqrt(2)`, and it is never
/// below either operand. A cheap invariant that no single-value test encodes.
#[test]
fn the_result_is_bracketed_by_its_operands() {
    let mut vals: Vec<f32> = Vec::new();
    for e in (-140..=120).step_by(4) {
        vals.push((2.0f64).powi(e) as f32 * 1.3);
    }

    for_each_tier!(|P, ti| {
        for &a in &vals {
            for &b in &vals {
                let got = f32x8::splat(a).hypot_p::<P>(f32x8::splat(b)).into_array()[0];
                let m = a.abs().max(b.abs());
                assert!(
                    got >= m * (1.0 - 1e-6),
                    "{}: hypot({a}, {b}) = {got} is below max operand {m}",
                    TIER_NAMES[ti]
                );
                assert!(
                    got <= m * core::f32::consts::SQRT_2 * (1.0 + 1e-6),
                    "{}: hypot({a}, {b}) = {got} exceeds max*sqrt(2) = {}",
                    TIER_NAMES[ti],
                    m * core::f32::consts::SQRT_2
                );
            }
        }
    });
}

// ---------------------------------------------------------------------------
// 5. Lane independence. A kernel that reads a cross-lane value would pass every
//    splatted test above and still be wrong in production.
// ---------------------------------------------------------------------------

#[test]
fn lanes_do_not_influence_each_other() {
    let probe: [f32; 8] = [3.0, 1e-40, 1e30, 0.0, f32::NAN, f32::INFINITY, f32::MAX, 1.4];
    let other: [f32; 8] = [4.0, 1e-40, 1e30, 0.0, 1.0, 1.0, f32::MIN_POSITIVE, 2.7];

    for_each_tier!(|P, ti| {
        let packed = f32x8::new(probe).hypot_p::<P>(f32x8::new(other)).into_array();

        for i in 0..8 {
            let alone = f32x8::splat(probe[i]).hypot_p::<P>(f32x8::splat(other[i])).into_array()[0];
            assert_eq!(
                packed[i].to_bits(),
                alone.to_bits(),
                "{}: lane {i} hypot({}, {}) = {} packed but {} alone",
                TIER_NAMES[ti],
                probe[i],
                other[i],
                packed[i],
                alone
            );
        }
    });
}

// ---------------------------------------------------------------------------
// 6. hypot_n at several N, and inv_hypot_n.
// ---------------------------------------------------------------------------

#[test]
fn hypot_n_agrees_with_a_scalar_reference() {
    fn reference(vals: &[f64]) -> f64 {
        // Scaled explicitly so the reference itself cannot overflow.
        let m = vals.iter().fold(0.0f64, |a, b| a.max(b.abs()));
        if m == 0.0 || !m.is_finite() {
            return if m.is_nan() { f64::NAN } else { m };
        }
        let s: f64 = vals.iter().map(|v| (v / m) * (v / m)).sum();
        m * s.sqrt()
    }

    let sets: [&[f32]; 7] = [
        &[3.0, 4.0],
        &[1.0, 2.0, 2.0],
        &[1e30, 1e30, 1e30],
        &[1e-40, 1e-40, 1e-40, 1e-40],
        &[f32::MAX, 1.0, 1.0],
        &[0.0, 0.0, 0.0],
        &[1.0, 1e-20, 1e20, 5.0],
    ];

    for_each_tier!(|P, ti| {
        for set in &sets {
            let want = reference(&set.iter().map(|&v| v as f64).collect::<Vec<_>>()) as f32;

            let got = match set.len() {
                2 => f32x8::splat(set[0]).hypot_p::<P>(f32x8::splat(set[1])),
                3 => f32x8::hypot_n_p::<P, 3>([f32x8::splat(set[0]), f32x8::splat(set[1]), f32x8::splat(set[2])]),
                4 => f32x8::hypot_n_p::<P, 4>([
                    f32x8::splat(set[0]),
                    f32x8::splat(set[1]),
                    f32x8::splat(set[2]),
                    f32x8::splat(set[3]),
                ]),
                _ => unreachable!(),
            }
            .into_array()[0];

            assert!(
                ulps_f32(got, want) <= 8.0,
                "{}: hypot_n({set:?}) = {got}, want {want}",
                TIER_NAMES[ti]
            );
        }
    });
}

#[test]
fn hypot_n_of_one_and_zero_elements() {
    for_each_tier!(|P, ti| {
        // N == 1 is |x|.
        for &x in &[3.0f32, -3.0, 0.0, -0.0, 1e-40, f32::MAX] {
            let got = f32x8::hypot_n_p::<P, 1>([f32x8::splat(x)]).into_array()[0];
            assert_eq!(
                got.to_bits(),
                x.abs().to_bits(),
                "{}: hypot_n<1>({x}) = {got}",
                TIER_NAMES[ti]
            );
        }

        // N == 0 is the empty norm, 0.
        let got = f32x8::hypot_n_p::<P, 0>([]).into_array()[0];
        assert_eq!(got, 0.0, "{}: hypot_n<0>() = {got}", TIER_NAMES[ti]);
    });
}

#[test]
fn inv_hypot_n_is_the_reciprocal_of_hypot_n() {
    let sets: [[f32; 3]; 6] = [
        [3.0, 4.0, 0.0],
        [1.0, 1.0, 1.0],
        [1e20, 1e20, 1e20],
        [1e-30, 1e-30, 1e-30],
        [1.0, 1e-10, 1e10],
        [f32::MAX, f32::MAX, 0.0],
    ];

    for_each_tier!(|P, ti| {
        for set in &sets {
            let v = [f32x8::splat(set[0]), f32x8::splat(set[1]), f32x8::splat(set[2])];
            let ih = f32x8::inv_hypot_n_p::<P, 3>(v).into_array()[0];

            // Against a float64 reference, NOT against `1.0 / hypot_n`.
            //
            // `inv_hypot_n` has a genuinely WIDER output range than the reciprocal of
            // `hypot_n`, and that is by construction rather than by accident: it scales
            // before taking the root, so it never forms the norm itself. Measured:
            // `inv_hypot_n([FLT_MAX, FLT_MAX, 0])` returns 2.078137e-39, the correct and
            // perfectly representable answer, while `hypot_n` of the same input overflows
            // to `inf` because 4.8e38 does not fit. Asserting `1.0 / h` would demand that
            // the inverse throw away a right answer to match a saturated one.
            let n: f64 = set.iter().map(|&x| (x as f64) * (x as f64)).sum();
            let want = if n == 0.0 {
                f32::INFINITY
            } else {
                (1.0 / n.sqrt()) as f32
            };

            if want.is_infinite() {
                assert!(
                    ih.is_infinite(),
                    "{}: inv_hypot_n({set:?}) = {ih} for a zero norm",
                    TIER_NAMES[ti]
                );
                continue;
            }

            let rel = ((ih - want) / want).abs();
            assert!(
                rel <= INV_REL_TOL,
                "{}: inv_hypot_n({set:?}) = {ih}, want {want} (rel {rel})",
                TIER_NAMES[ti]
            );
        }
    });
}

/// A 3-vector normalized by `inv_hypot_n` must come out unit length, which is what the
/// function is actually used for.
#[test]
fn inv_hypot_n_normalizes_to_unit_length() {
    let vecs: [[f32; 3]; 6] = [
        [3.0, 4.0, 12.0],
        [1e30, -2e30, 0.5e30],
        [1e-38, 2e-38, 3e-38],
        [1.0, 0.0, 0.0],
        [-1.0, -1.0, -1.0],
        [1e20, 1e-20, 1.0],
    ];

    for_each_tier!(|P, ti| {
        for v in &vecs {
            let vv = [f32x8::splat(v[0]), f32x8::splat(v[1]), f32x8::splat(v[2])];
            let inv = f32x8::inv_hypot_n_p::<P, 3>(vv).into_array()[0];

            let n = [v[0] * inv, v[1] * inv, v[2] * inv];
            let len = ((n[0] as f64).powi(2) + (n[1] as f64).powi(2) + (n[2] as f64).powi(2)).sqrt();

            assert!(
                (len - 1.0).abs() <= 2e-2,
                "{}: normalizing {v:?} gave length {len}",
                TIER_NAMES[ti]
            );
        }
    });
}

// ---------------------------------------------------------------------------
// 7. Cross-backend agreement. The kernel is shared, but the lowering is not.
// ---------------------------------------------------------------------------

#[test]
fn every_backend_agrees_with_the_scalar_backend() {
    use thermite::backend::scalar::Scalar;
    use thermite::backend::{x86_v1::X86V1, x86_v2::X86V2};
    use thermite::simd::Simd;

    type S1 = Vector<<Scalar as Simd>::f32x4>;
    type V1 = Vector<<X86V1 as Simd>::f32x4>;
    type V2 = Vector<<X86V2 as Simd>::f32x8>;

    let cases: [(f32, f32); 12] = [
        (3.0, 4.0),
        (0.0, 0.0),
        (1e30, 1e30),
        (1e-40, 1e-40),
        (f32::MAX, f32::MAX),
        (f32::from_bits(1), f32::from_bits(1)),
        (9.5e-23, 3.2e20),
        (1.0, f32::NAN),
        (f32::NAN, 1.0),
        (f32::INFINITY, 1.0),
        (1.0, f32::INFINITY),
        (-5.0, 12.0),
    ];

    for &(x, y) in &cases {
        let want = libm::hypotf(x, y);

        let s = S1::splat(x).hypot_p::<Precision>(S1::splat(y)).into_array()[0];
        let v1 = V1::splat(x).hypot_p::<Precision>(V1::splat(y)).into_array()[0];
        let v2 = V2::splat(x).hypot_p::<Precision>(V2::splat(y)).into_array()[0];
        let v3 = f32x8::splat(x).hypot_p::<Precision>(f32x8::splat(y)).into_array()[0];

        for (name, got) in [("scalar", s), ("x86_v1", v1), ("x86_v2", v2), ("x86_v3", v3)] {
            assert!(
                ulps_f32(got, want) <= ULP_TOL,
                "{name}: hypot({x}, {y}) = {got}, want {want}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 8. The invariance claim itself, stated as a test.
// ---------------------------------------------------------------------------

/// Every tier must return the SAME BITS. This is the property the rewrite
/// bought, and it is worth asserting directly rather than inferring it from seven
/// tolerance checks passing.
///
/// `Reference` is excluded: it calls scalar `libm` per lane by contract, so it is a
/// different implementation rather than a different policy.
#[test]
fn every_non_reference_tier_returns_identical_bits() {
    let mut vals: Vec<f32> = Vec::new();
    for e in (-149..=127).step_by(5) {
        vals.push((2.0f64).powi(e) as f32);
        vals.push((2.0f64).powi(e) as f32 * 1.7320508);
    }
    vals.extend([0.0, -0.0, f32::MAX, f32::MIN_POSITIVE]);

    for &a in &vals {
        for &b in &vals {
            let ultra = f32x8::splat(a)
                .hypot_p::<UltraPerformance>(f32x8::splat(b))
                .into_array()[0];
            let high = f32x8::splat(a).hypot_p::<HighPerformance>(f32x8::splat(b)).into_array()[0];
            let perf = f32x8::splat(a).hypot_p::<Performance>(f32x8::splat(b)).into_array()[0];
            let size = f32x8::splat(a).hypot_p::<Size>(f32x8::splat(b)).into_array()[0];
            let prec = f32x8::splat(a).hypot_p::<Precision>(f32x8::splat(b)).into_array()[0];

            for (name, got) in [
                ("UltraPerformance", ultra),
                ("HighPerformance", high),
                ("Size", size),
                ("Precision", prec),
            ] {
                assert_eq!(
                    got.to_bits(),
                    perf.to_bits(),
                    "{name} disagrees with Performance on hypot({a}, {b}): {got} vs {perf}"
                );
            }
        }
    }
}
