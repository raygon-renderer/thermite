//! Chebyshev series summation: `chebyshev_n::<K, N>` over all four kinds.
//!
//! The reference is a compensated (Neumaier) forward sum of `c_k * P_k(x)`, which shares
//! no structure with Clenshaw's backward recurrence and carries roughly twice the working
//! precision of the plain sum, so the tolerances below measure the kernel rather than the
//! oracle's own drift.
//!
//! Coverage is organized around what can independently go wrong: the `P_1` selection per
//! kind (checked against closed forms in theta, which a value table cannot distinguish
//! from a wrong recurrence), the shared `b_k` loop (checked against the forward sum), the
//! short-series shortcuts at `N = 1` and `N = 2` (which skip the loop entirely), and the
//! endpoint behaviour of the policy-gated Reinsch form.
//!
//! The endpoint test asserts on the error *envelope* over a grid, not pointwise. Reinsch
//! bounds the worst case 2.5x to 17x tighter but is not pointwise dominant. Individual
//! arguments land either way, and a pointwise assertion would be flaky by construction.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]

use thermite::math::policy::policies::{Performance, Precision};
use thermite::prelude::*;
use thermite_special::SpecialMathWithPolicy;

type D = Vector<f64>;
type FV = Vector<f32>;

/// `P_1(x)` per kind. All four share `P_0 = 1` and `P_{k+1} = 2x P_k - P_{k-1}`.
fn p1(kind: usize, x: f64) -> f64 {
    match kind {
        1 => x,
        2 => 2.0 * x,
        3 => 2.0 * x - 1.0,
        4 => 2.0 * x + 1.0,
        _ => unreachable!(),
    }
}

// Minimal double-double, about 106 bits. Compensating only the accumulation is not
// enough here: `P_{k+1} = 2x P_k - P_{k-1}` is itself the cancellation-prone step, and an
// oracle that runs it in plain f64 inherits exactly the error it is supposed to measure.
// That version reported Reinsch as the worse algorithm at K=2, which the 60-digit mpmath
// probe in `bin/precision_audit` contradicts.

type Dd = (f64, f64);

fn two_sum(a: f64, b: f64) -> Dd {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

fn quick_two_sum(a: f64, b: f64) -> Dd {
    let s = a + b;
    (s, b - (s - a))
}

fn two_prod(a: f64, b: f64) -> Dd {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

fn dd_add(a: Dd, b: Dd) -> Dd {
    let (s, e) = two_sum(a.0, b.0);
    quick_two_sum(s, e + a.1 + b.1)
}

fn dd_mul_f64(a: Dd, b: f64) -> Dd {
    let (p, e) = two_prod(a.0, b);
    quick_two_sum(p, e + a.1 * b)
}

/// Forward sum of the series with both the recurrence and the accumulation in
/// double-double, independent of Clenshaw in structure as well as in precision.
fn reference(kind: usize, coeffs: &[f64], x: f64) -> f64 {
    // P_1 exactly: 2x is exact for any finite x, and two_sum captures the rest.
    let mut cur: Dd = match kind {
        1 => (x, 0.0),
        2 => (2.0 * x, 0.0),
        3 => two_sum(2.0 * x, -1.0),
        4 => two_sum(2.0 * x, 1.0),
        _ => unreachable!(),
    };
    let mut prev: Dd = (1.0, 0.0);
    let mut sum: Dd = (coeffs[0], 0.0);

    for &c in &coeffs[1..] {
        sum = dd_add(sum, dd_mul_f64(cur, c));
        let next = dd_add(dd_mul_f64(cur, 2.0 * x), (-prev.0, -prev.1));
        prev = cur;
        cur = next;
    }

    sum.0 + sum.1
}

fn rel_err(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs().max(1.0)
}

/// `sum |c_k P_k(x)|`, the scale the summation has to cancel through, and therefore the
/// error any algorithm is entitled to. Needed for the f32 checks, where a flat tolerance
/// would be a false alarm exactly on the adversarial coefficient sets.
fn condition(kind: usize, coeffs: &[f64], x: f64) -> f64 {
    let (mut prev, mut cur) = (1.0, p1(kind, x));
    let mut sum = coeffs[0].abs();
    for &c in &coeffs[1..] {
        sum += (c * cur).abs();
        (prev, cur) = (cur, 2.0 * x * cur - prev);
    }
    sum
}

/// Geometrically decaying, alternating: what a minimax or least-squares fit produces.
fn decay<const N: usize>() -> [f64; N] {
    let mut c = [0.0; N];
    for (k, slot) in c.iter_mut().enumerate() {
        *slot = if k % 2 == 0 { 1.0 } else { -1.0 } * 0.7f64.powi(k as i32);
    }
    c
}

/// A grid that includes both endpoints exactly and approaches each of them by halving.
fn endpoint_grid() -> Vec<f64> {
    let mut xs = vec![1.0, -1.0];
    for j in 1..=40 {
        let e = 2f64.powi(-j);
        xs.push(1.0 - e);
        xs.push(-1.0 + e);
    }
    xs
}

const N: usize = 24;

/// Narrow a coefficient set to f32. The reference stays in f64 throughout, and only the
/// coefficients the f32 kernel actually receives are rounded.
fn c32(c: &[f64; N]) -> [f32; N] {
    let mut out = [0.0f32; N];
    for (o, &v) in out.iter_mut().zip(c.iter()) {
        *o = v as f32;
    }
    out
}

#[test]
fn matches_compensated_forward_sum_on_all_kinds() {
    let c = decay::<N>();
    let xs: Vec<f64> = (-20..=20).map(|i| i as f64 / 20.0).collect();

    for &x in &xs {
        let v = D::splat(x);

        macro_rules! check {
            ($k:literal) => {{
                let want = reference($k, &c, x);
                let fast = v.chebyshev_n_p::<Performance, $k, N>(&c).extract::<0>();
                let best = v.chebyshev_n_p::<Precision, $k, N>(&c).extract::<0>();
                assert!(
                    rel_err(fast, want) <= 1e-14,
                    "K={} clenshaw at x={x}: got {fast}, want {want}",
                    $k
                );
                assert!(
                    rel_err(best, want) <= 1e-14,
                    "K={} reinsch at x={x}: got {best}, want {want}",
                    $k
                );
            }};
        }

        check!(1);
        check!(2);
        check!(3);
        check!(4);
    }
}

#[test]
fn kinds_match_their_closed_forms_in_theta() {
    // A single non-zero coefficient isolates P_k itself:
    //   T_k(cos t) = cos(k t)
    //   U_k(cos t) = sin((k+1) t) / sin(t)
    //   V_k(cos t) = cos((k + 1/2) t) / cos(t / 2)
    //   W_k(cos t) = sin((k + 1/2) t) / sin(t / 2)
    // These distinguish the four kinds from one another. A table of values at one kind
    // would pass just as well against a wrong P_1.
    const DEG: usize = 7;
    let mut c = [0.0; DEG + 1];
    c[DEG] = 1.0;

    for i in 1..=60 {
        let t = core::f64::consts::PI * i as f64 / 61.0;
        let v = D::splat(t.cos());
        let k = DEG as f64;

        let got_t = v.chebyshev_n_p::<Precision, 1, { DEG + 1 }>(&c).extract::<0>();
        let got_u = v.chebyshev_n_p::<Precision, 2, { DEG + 1 }>(&c).extract::<0>();
        let got_v = v.chebyshev_n_p::<Precision, 3, { DEG + 1 }>(&c).extract::<0>();
        let got_w = v.chebyshev_n_p::<Precision, 4, { DEG + 1 }>(&c).extract::<0>();

        assert!(rel_err(got_t, (k * t).cos()) <= 1e-13, "T_{DEG} at t={t}");
        assert!(
            rel_err(got_u, ((k + 1.0) * t).sin() / t.sin()) <= 1e-13,
            "U_{DEG} at t={t}"
        );
        assert!(
            rel_err(got_v, ((k + 0.5) * t).cos() / (t / 2.0).cos()) <= 1e-13,
            "V_{DEG} at t={t}"
        );
        assert!(
            rel_err(got_w, ((k + 0.5) * t).sin() / (t / 2.0).sin()) <= 1e-13,
            "W_{DEG} at t={t}"
        );
    }
}

#[test]
fn short_series_skip_the_recurrence() {
    // N = 1 and N = 2 return before the loop, so they need their own coverage.
    for &x in &[-1.0, -0.5, 0.0, 0.25, 1.0] {
        let v = D::splat(x);

        let one = [2.5];
        assert_eq!(v.chebyshev_n_p::<Precision, 1, 1>(&one).extract::<0>(), 2.5);
        assert_eq!(v.chebyshev_n_p::<Precision, 4, 1>(&one).extract::<0>(), 2.5);

        let two = [2.5, -1.25];
        for kind in 1..=4 {
            let want = 2.5 - 1.25 * p1(kind, x);
            let got = match kind {
                1 => v.chebyshev_n_p::<Precision, 1, 2>(&two).extract::<0>(),
                2 => v.chebyshev_n_p::<Precision, 2, 2>(&two).extract::<0>(),
                3 => v.chebyshev_n_p::<Precision, 3, 2>(&two).extract::<0>(),
                _ => v.chebyshev_n_p::<Precision, 4, 2>(&two).extract::<0>(),
            };
            assert!(rel_err(got, want) <= 1e-15, "K={kind} N=2 at x={x}");
        }
    }
}

#[test]
fn precision_policy_bounds_the_endpoint_error_more_tightly() {
    // Non-decaying coefficients of alternating sign: the regime where the recurrence's
    // own cancellation, rather than the series' conditioning, sets the error. A fitted
    // spectrum decays too fast to show the effect (see the control test below).
    let mut c = [0.0; N];
    for (k, slot) in c.iter_mut().enumerate() {
        *slot = if k % 2 == 0 { 1.0 } else { -1.0 };
    }

    let xs = endpoint_grid();

    macro_rules! envelope {
        ($k:literal) => {{
            let (mut worst_fast, mut worst_best) = (0.0f64, 0.0f64);
            for &x in &xs {
                let v = D::splat(x);
                let want = reference($k, &c, x);
                worst_fast = worst_fast.max(rel_err(v.chebyshev_n_p::<Performance, $k, N>(&c).extract::<0>(), want));
                worst_best = worst_best.max(rel_err(v.chebyshev_n_p::<Precision, $k, N>(&c).extract::<0>(), want));
            }
            assert!(
                worst_best < worst_fast,
                "K={}: Reinsch envelope {worst_best:e} should beat Clenshaw's {worst_fast:e}",
                $k
            );
            assert!(
                worst_best <= 1e-13,
                "K={}: Reinsch envelope {worst_best:e} too loose",
                $k
            );
        }};
    }

    envelope!(1);
    envelope!(2);
    envelope!(3);
    envelope!(4);
}

#[test]
fn lanes_carrying_different_arguments_stay_independent() {
    // Everything above runs one lane. The coefficient array is scalar but x is not, so a
    // kernel that reduced across lanes anywhere in the recurrence would pass all of it.
    use thermite::backend::scalar::Scalar;
    type D4 = thermite::simd::f64x4<Scalar>;

    let c = decay::<N>();
    // Deliberately straddling zero, which is where the branchless Reinsch sign selection
    // has to disagree between lanes: a scalar `if x >= 0` would give the wrong answer for
    // half of these.
    let xs = [-0.99999, -0.25, 0.25, 0.99999];

    let got_fast = D4::new(xs).chebyshev_n_p::<Performance, 1, N>(&c);
    let got_best = D4::new(xs).chebyshev_n_p::<Precision, 1, N>(&c);

    for (lane, &x) in xs.iter().enumerate() {
        let want = reference(1, &c, x);
        assert!(
            rel_err(got_fast.as_slice()[lane], want) <= 1e-14,
            "clenshaw lane {lane} at x={x}"
        );
        assert!(
            rel_err(got_best.as_slice()[lane], want) <= 1e-14,
            "reinsch lane {lane} at x={x}"
        );
        // And each lane must equal what that argument produces on its own.
        let solo = D::splat(x).chebyshev_n_p::<Precision, 1, N>(&c).extract::<0>();
        assert_eq!(
            got_best.as_slice()[lane],
            solo,
            "lane {lane} differs from the solo result"
        );
    }
}

#[test]
fn both_endpoint_forms_are_exercised_within_one_vector() {
    // The sharper version of the test above: adversarial coefficients, and every lane
    // pinned right up against an endpoint, alternating sides. If the sign selection were
    // ever hoisted out of the lanes this would fail on half of them.
    use thermite::backend::scalar::Scalar;
    type D4 = thermite::simd::f64x4<Scalar>;

    let mut c = [0.0; N];
    for (k, slot) in c.iter_mut().enumerate() {
        *slot = if k % 2 == 0 { 1.0 } else { -1.0 };
    }

    for j in 1..=30 {
        let e = 2f64.powi(-j);
        let xs = [1.0 - e, -1.0 + e, -(1.0 - e), 1.0 - e * 0.5];
        let got = D4::new(xs).chebyshev_n_p::<Precision, 1, N>(&c);

        for (lane, &x) in xs.iter().enumerate() {
            let want = reference(1, &c, x);
            let bound = f64::EPSILON * condition(1, &c, x) * 64.0;
            assert!(
                (got.as_slice()[lane] - want).abs() <= bound,
                "lane {lane} at x={x}: got {}, want {want}",
                got.as_slice()[lane]
            );
        }
    }
}

// --- f32 ---

#[test]
fn f32_matches_the_reference_on_all_kinds() {
    let c = decay::<N>();
    let xs: Vec<f64> = (-16..=16).map(|i| i as f64 / 16.0).collect();

    for &x in &xs {
        let v = FV::splat(x as f32);

        macro_rules! check {
            ($k:literal) => {{
                let want = reference($k, &c, x);
                let bound = f32::EPSILON as f64 * condition($k, &c, x) * 8.0;
                for (label, got) in [
                    (
                        "clenshaw",
                        v.chebyshev_n_p::<Performance, $k, N>(&c32(&c)).extract::<0>(),
                    ),
                    (
                        "reinsch",
                        v.chebyshev_n_p::<Precision, $k, N>(&c32(&c)).extract::<0>(),
                    ),
                ] {
                    let err = (got as f64 - want).abs();
                    assert!(
                        err <= bound,
                        "f32 K={} {label} at x={x}: err {err:e} > {bound:e}",
                        $k
                    );
                }
            }};
        }

        check!(1);
        check!(2);
        check!(3);
        check!(4);
    }
}

/// An f32-native endpoint grid: the representable neighbours of `+-1`, then halvings only
/// as far as they stay distinct.
///
/// The f64 grid is the wrong instrument here and quietly reports a null result. `1 - 2^-j`
/// rounds to exactly `1.0` in binary32 for every `j >= 24`, so two thirds of it lands on
/// the endpoint itself, where `step = 2(x - s)` is exactly zero, the endpoint form
/// degenerates into a plain running sum, and both policies agree to the bit, while the
/// arguments that actually stress the recurrence go unsampled.
fn f32_endpoint_grid() -> Vec<f32> {
    let mut xs = vec![1.0f32, -1.0f32];
    let one = 1.0f32.to_bits();
    for k in 1..=48u32 {
        let v = f32::from_bits(one - k);
        xs.push(v);
        xs.push(-v);
    }
    for j in 1..=23 {
        let e = 2f32.powi(-j);
        xs.push(1.0 - e);
        xs.push(-1.0 + e);
    }
    xs
}

#[test]
fn f32_precision_policy_also_tightens_the_endpoint_envelope() {
    // `ps.rs` opts f32 into Reinsch through its own override, separate from `pd.rs`, so a
    // lost override here is invisible to the f64 test. Measured on the grid above, f32
    // gains 2.6x to 13.5x across the four kinds, the same order as f64, not the ~1% an
    // f64 grid appears to show.
    //
    // The margin depends on how the scalar backend lowers `mul_adde`: without a baseline
    // FMA (x86 default) the worst kind still gains 2.78x, but where HAS_NATIVE_FMA fuses it
    // (aarch64, x86 with -C target-feature=+fma) the single fused rounding helps Clenshaw's
    // `2x*b + (c - b_2)` step more than it helps Reinsch, and K=1 narrows to 1.72x. Both
    // lowerings are deterministic, so 1.5x sits inside both measured margins. A lost
    // REINSCH override reads as ratio 1.0 and is caught by the `differ > 0` assert anyway.
    let mut c = [0.0; N];
    for (k, slot) in c.iter_mut().enumerate() {
        *slot = if k % 2 == 0 { 1.0 } else { -1.0 };
    }
    let c32v = c32(&c);
    let xs = f32_endpoint_grid();

    macro_rules! envelope {
        ($k:literal) => {{
            let (mut worst_fast, mut worst_best) = (0.0f64, 0.0f64);
            let mut differ = 0;
            for &x in &xs {
                let v = FV::splat(x);
                let fast = v.chebyshev_n_p::<Performance, $k, N>(&c32v).extract::<0>();
                let best = v.chebyshev_n_p::<Precision, $k, N>(&c32v).extract::<0>();
                if fast.to_bits() != best.to_bits() {
                    differ += 1;
                }
                let want = reference($k, &c, x as f64);
                worst_fast = worst_fast.max((fast as f64 - want).abs());
                worst_best = worst_best.max((best as f64 - want).abs());
            }

            assert!(
                differ > 0,
                "K={}: f32 policies agreed everywhere - is ps.rs still passing REINSCH?",
                $k
            );
            assert!(
                worst_best * 1.5 < worst_fast,
                "K={}: f32 Reinsch envelope {worst_best:e} should clearly beat Clenshaw's {worst_fast:e}",
                $k
            );
        }};
    }

    envelope!(1);
    envelope!(2);
    envelope!(3);
    envelope!(4);
}

#[test]
fn fitted_spectra_are_accurate_at_the_endpoints_under_either_policy() {
    // The control for the test above: geometric decay is the common case, and it must
    // stay tight at the endpoints without needing the Best-policy path.
    let c = decay::<N>();

    for &x in &endpoint_grid() {
        let v = D::splat(x);
        for kind in 1..=4 {
            let want = reference(kind, &c, x);
            let got = match kind {
                1 => v.chebyshev_n_p::<Performance, 1, N>(&c).extract::<0>(),
                2 => v.chebyshev_n_p::<Performance, 2, N>(&c).extract::<0>(),
                3 => v.chebyshev_n_p::<Performance, 3, N>(&c).extract::<0>(),
                _ => v.chebyshev_n_p::<Performance, 4, N>(&c).extract::<0>(),
            };
            assert!(rel_err(got, want) <= 1e-14, "K={kind} at x={x}: got {got}, want {want}");
        }
    }
}
