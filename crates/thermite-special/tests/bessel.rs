//! Correctness gate for the modified Bessel functions `I_0` and `I_1`, scaled and unscaled,
//! driven through the public `SpecialMath` entries, which are also the dispatched paths.
//!
//! The scaled and unscaled forms are checked against _separately generated_ reference
//! columns rather than against each other, because they are not built from each other: over
//! most of the range the tables are natively the scaled quantity and the unscaled form is the
//! one paying for an exponential. Testing `i0e * exp(x) == i0` alone would pass even if both
//! shared a wrong polynomial, and would say nothing at all past `x = 714`, where the unscaled
//! value has left f64 entirely and the scaled one is still perfectly ordinary.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]
#![allow(clippy::excessive_precision)]

use thermite::Vector;
use thermite::math::policy::policies::{Performance, Precision};
use thermite::prelude::*;
use thermite_special::bessel::{I, J, K, Scaled, Y};
use thermite_special::{BesselOrder, SpecialMathWithPolicy};

type V = Vector<f64>;
type Vf = Vector<f32>;

include!("bessel_ref/table.rs");

fn i0(x: f64) -> f64 {
    V::splat(x).bessel_n_p::<Precision, I, 0>().extract::<0>()
}
fn i1(x: f64) -> f64 {
    V::splat(x).bessel_n_p::<Precision, I, 1>().extract::<0>()
}
fn i0e(x: f64) -> f64 {
    V::splat(x).bessel_n_p::<Precision, Scaled<I>, 0>().extract::<0>()
}
fn i1e(x: f64) -> f64 {
    V::splat(x).bessel_n_p::<Precision, Scaled<I>, 1>().extract::<0>()
}

fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        return if got == 0.0 { 0.0 } else { f64::INFINITY };
    }
    ((got - want) / want).abs()
}

#[test]
fn bessel_i_matches_mpmath_f64() {
    let mut worst_u = 0.0f64;
    let mut worst_s = 0.0f64;
    let mut bad = 0usize;
    for &(nu, x, want, want_scaled) in BESSEL_I.iter() {
        let (got, got_scaled) = if nu == 0 { (i0(x), i0e(x)) } else { (i1(x), i1e(x)) };

        // The unscaled column is only meaningful where mpmath's value survived the cast.
        //
        // Its gate grows with `x`. That is not slack, it is `exp` itself. Argument
        // reduction can only place `x - k ln2` to within `ulp(x)`, and `exp` turns an
        // absolute argument error straight into a relative output error, so any `e^x` costs
        // about `x * eps/2` relative no matter how good the kernel above it is. Measured
        // 2.3e-14 at x = 500, against a scaled column that stays flat at 3e-15 across the
        // entire grid including x = 1e6. That gap is the scaled form's second reason to
        // exist, after range.
        if want.is_finite() {
            let tol = 4e-15 + 1e-16 * x.abs();
            let e = rel(got, want);
            if e > tol {
                bad += 1;
                std::println!("I_{nu}({x}): got {got}, want {want}, rel {e:e} (tol {tol:e})");
            }
            worst_u = worst_u.max(e);
        }

        let e = rel(got_scaled, want_scaled);
        if e > 4e-15 {
            bad += 1;
            std::println!("e^-x I_{nu}({x}): got {got_scaled}, want {want_scaled}, rel {e:e}");
        }
        worst_s = worst_s.max(e);
    }
    std::println!(
        "f64: worst unscaled {worst_u:e}, worst scaled {worst_s:e} over {} rows",
        BESSEL_I.len()
    );
    assert!(bad == 0, "{bad} rows over the gate");
}

#[test]
fn bessel_i_matches_mpmath_f32() {
    let mut worst = 0.0f32;
    let mut rows = 0usize;
    for &(nu, x, _, want_scaled) in BESSEL_I.iter() {
        // f32 carries the scaled form over the whole grid, but the unscaled one would overflow
        // at x = 92, which the dedicated overflow test covers instead.
        if want_scaled == 0.0 || (want_scaled as f32) == 0.0 {
            continue;
        }
        let xf = Vf::splat(x as f32);
        let got = if nu == 0 {
            xf.bessel_n_p::<Precision, Scaled<I>, 0>()
        } else {
            xf.bessel_n_p::<Precision, Scaled<I>, 1>()
        }
        .extract::<0>();
        let want = want_scaled as f32;
        let e = ((got - want) / want).abs();
        assert!(e <= 3e-6, "f32 e^-x I_{nu}({x}): got {got}, want {want}, rel {e:e}");
        worst = worst.max(e);
        rows += 1;
    }
    std::println!("f32 scaled: worst rel {worst:e} over {rows} rows");
}

#[test]
fn bessel_i_parity_is_exact() {
    // I_0 is even and I_1 odd, and the kernel gets that from `abs` and `copysign` rather
    // than from the polynomial, so it must hold to the last bit, not to a tolerance.
    for &x in &[1e-8, 0.001, 0.5, 3.0, 7.7, 7.75, 7.8, 20.0, 60.0, 500.0, 600.0] {
        assert!(
            i0(x).to_bits() == i0(-x).to_bits(),
            "I_0 not even at {x}: {} vs {}",
            i0(x),
            i0(-x)
        );
        assert!(
            i1(x).to_bits() == (-i1(-x)).to_bits(),
            "I_1 not odd at {x}: {} vs {}",
            i1(x),
            i1(-x)
        );
        assert!(i0e(x).to_bits() == i0e(-x).to_bits(), "scaled I_0 not even at {x}");
        assert!(i1e(x).to_bits() == (-i1e(-x)).to_bits(), "scaled I_1 not odd at {x}");
    }
}

#[test]
fn bessel_i_scaled_survives_where_unscaled_overflows() {
    // The whole reason the scaled form is a separate primitive rather than a wrapper.
    assert!(i0(800.0).is_infinite(), "I_0(800) should overflow, got {}", i0(800.0));
    assert!(i1(800.0).is_infinite(), "I_1(800) should overflow, got {}", i1(800.0));

    let s = i0e(800.0);
    assert!(
        rel(s, 0.0141069450058692) <= 1e-14,
        "e^-800 I_0(800): got {s}, want 0.0141069450058692"
    );
    // ... and keeps going far past it.
    assert!(
        i0e(1e6).is_finite() && i0e(1e6) > 0.0,
        "scaled I_0 died at 1e6: {}",
        i0e(1e6)
    );

    // Building it the naive way is exactly what does not work.
    assert!(
        (i0(800.0) * (-800.0f64).exp()).is_nan(),
        "inf * 0 should be NaN - if this ever stops holding, the scaled form's reason to exist changed"
    );
}

#[test]
fn bessel_i_derivative_identity() {
    // I_0' = I_1 exactly, by central differences on the value form. This crosses the two
    // kernels, which share no coefficients: I_0's small fit is in `1 + a P(a)` and I_1's is
    // the nested `(x/2)(1 + a(1/2 + a P(a)))`.
    for &x in &[0.25f64, 1.0, 3.0, 6.0, 9.0, 15.0, 40.0] {
        // 1e-5 leaves 1.2e-9 of pure central-difference truncation at x = 9, which is the
        // difference formula's own error and says nothing about the kernel. 1e-6 puts
        // truncation (~h^2) at 1e-13 and roundoff (~eps/h) at 2e-11.
        let h = 1e-6 * x.max(1.0);
        let d = (i0(x + h) - i0(x - h)) / (2.0 * h);
        let e = rel(d, i1(x));
        assert!(e <= 1e-9, "d/dx I_0({x}) = {d}, I_1 = {}, rel {e:e}", i1(x));
    }
}

#[test]
fn bessel_i_seam_is_continuous() {
    // The 7.75 handover joins two completely different approximations. A discontinuity there
    // is the classic region-split defect, and hides from a coarse sweep.
    let below = i0(7.75f64.next_down());
    let above = i0(7.75);
    assert!(
        rel(below, above) <= 1e-14,
        "I_0 jumps across the 7.75 seam: {below} vs {above}"
    );
    let below = i1(7.75f64.next_down());
    let above = i1(7.75);
    assert!(
        rel(below, above) <= 1e-14,
        "I_1 jumps across the 7.75 seam: {below} vs {above}"
    );

    // The far seam at 500 changes both the polynomial and how the exponential is assembled.
    let below = i0e(500.0f64.next_down());
    let above = i0e(500.0);
    assert!(
        rel(below, above) <= 1e-14,
        "scaled I_0 jumps across the 500 seam: {below} vs {above}"
    );
}

#[test]
fn bessel_i_default_policy_tracks_precision() {
    // `Performance` is `PrecisionPolicy::Average` (the default every non-`_p` call takes),
    // so it needs its own gate. It shares the kernel here, differing only in how `exp`,
    // `sqrt` and the Horner chain are lowered.
    for &(nu, x, _, want_scaled) in BESSEL_I.iter() {
        if !(0.0..=200.0).contains(&x) {
            continue;
        }
        let v = V::splat(x);
        let got = if nu == 0 {
            v.bessel_n_p::<Performance, Scaled<I>, 0>()
        } else {
            v.bessel_n_p::<Performance, Scaled<I>, 1>()
        }
        .extract::<0>();
        let e = rel(got, want_scaled);
        assert!(
            e <= 1e-13,
            "Performance e^-x I_{nu}({x}): got {got}, want {want_scaled}, rel {e:e}"
        );
    }
}

include!("bessel_ref/table_n.rs");

/// Dispatch a const order from a runtime one, for table-driven tests only. Two macros
/// rather than one with a bool: a `literal` metavariable cannot be matched against `true`
/// once forwarded, and `:tt` here would only obscure what is a two-line duplication.
macro_rules! at_order_scaled {
    ($n:expr, $v:expr, $tier:ty) => {
        match $n {
            2 => $v.bessel_n_p::<$tier, Scaled<I>, 2>().extract::<0>(),
            3 => $v.bessel_n_p::<$tier, Scaled<I>, 3>().extract::<0>(),
            5 => $v.bessel_n_p::<$tier, Scaled<I>, 5>().extract::<0>(),
            8 => $v.bessel_n_p::<$tier, Scaled<I>, 8>().extract::<0>(),
            12 => $v.bessel_n_p::<$tier, Scaled<I>, 12>().extract::<0>(),
            20 => $v.bessel_n_p::<$tier, Scaled<I>, 20>().extract::<0>(),
            30 => $v.bessel_n_p::<$tier, Scaled<I>, 30>().extract::<0>(),
            50 => $v.bessel_n_p::<$tier, Scaled<I>, 50>().extract::<0>(),
            _ => unreachable!("order {} not in the table's grid", $n),
        }
    };
}

macro_rules! at_order_plain {
    ($n:expr, $v:expr, $tier:ty) => {
        match $n {
            2 => $v.bessel_n_p::<$tier, I, 2>().extract::<0>(),
            3 => $v.bessel_n_p::<$tier, I, 3>().extract::<0>(),
            5 => $v.bessel_n_p::<$tier, I, 5>().extract::<0>(),
            8 => $v.bessel_n_p::<$tier, I, 8>().extract::<0>(),
            12 => $v.bessel_n_p::<$tier, I, 12>().extract::<0>(),
            20 => $v.bessel_n_p::<$tier, I, 20>().extract::<0>(),
            30 => $v.bessel_n_p::<$tier, I, 30>().extract::<0>(),
            50 => $v.bessel_n_p::<$tier, I, 50>().extract::<0>(),
            _ => unreachable!("order {} not in the table's grid", $n),
        }
    };
}

#[test]
fn bessel_in_recurrence_matches_mpmath() {
    // The ratio recurrence, orders 2..50. The gate is the measured tier bound for `Best`
    // (5.19e-15 worst over N in 2..80, x in 0.01..700), with the same linear-in-x allowance
    // on the unscaled column that the closed forms need, because it inherits the same `exp`.
    let mut worst_s = 0.0f64;
    let mut worst_u = 0.0f64;
    let mut bad = 0usize;
    for &(n, x, want, want_scaled) in BESSEL_IN.iter() {
        let v = V::splat(x);
        let got_s = at_order_scaled!(n, v, Precision);
        let e = rel(got_s, want_scaled);
        if e > 2e-14 {
            bad += 1;
            std::println!("e^-x I_{n}({x}): got {got_s}, want {want_scaled}, rel {e:e}");
        }
        worst_s = worst_s.max(e);

        if want.is_finite() {
            let got_u = at_order_plain!(n, v, Precision);
            let tol = 2e-14 + 1e-16 * x;
            let e = rel(got_u, want);
            if e > tol {
                bad += 1;
                std::println!("I_{n}({x}): got {got_u}, want {want}, rel {e:e} (tol {tol:e})");
            }
            worst_u = worst_u.max(e);
        }
    }
    std::println!(
        "recurrence: worst scaled {worst_s:e}, worst unscaled {worst_u:e} over {} rows",
        BESSEL_IN.len()
    );
    assert!(bad == 0, "{bad} rows over the gate");
}

#[test]
fn bessel_in_joins_the_closed_forms() {
    // Order 2 by recurrence against the identity I_2 = I_0 - (2/x) I_1, which uses only the
    // closed forms, so a bug in the recurrence cannot hide behind a shared kernel.
    for &x in &[0.5f64, 1.0, 3.0, 7.0, 7.75, 9.0, 20.0, 60.0] {
        let want = i0(x) - (2.0 / x) * i1(x);
        let got = V::splat(x).bessel_n_p::<Precision, I, 2>().extract::<0>();
        assert!(rel(got, want) <= 1e-13, "I_2({x}): recurrence {got}, identity {want}");
    }
}

#[test]
fn bessel_in_parity_is_exact() {
    // Even order even, odd order odd. The recurrence must not smear that.
    for &x in &[0.5f64, 3.0, 9.0, 40.0] {
        let v = V::splat(x);
        let w = V::splat(-x);
        assert!(
            v.bessel_n_p::<Precision, I, 2>().extract::<0>().to_bits()
                == w.bessel_n_p::<Precision, I, 2>().extract::<0>().to_bits(),
            "I_2 not even at {x}"
        );
        let a = v.bessel_n_p::<Precision, I, 3>().extract::<0>();
        let b = w.bessel_n_p::<Precision, I, 3>().extract::<0>();
        assert!(a.to_bits() == (-b).to_bits(), "I_3 not odd at {x}: {a} vs {b}");
    }
}

#[test]
fn bessel_in_is_zero_at_the_origin() {
    // I_N(0) = 0 for N >= 1, and must fall out of the recurrence rather than a special
    // case: 2/x is +inf there, which drives every ratio to zero.
    for_each_order_at_zero();
}

fn for_each_order_at_zero() {
    let z = V::splat(0.0);
    assert!(z.bessel_n_p::<Precision, I, 2>().extract::<0>() == 0.0);
    assert!(z.bessel_n_p::<Precision, I, 5>().extract::<0>() == 0.0);
    assert!(z.bessel_n_p::<Precision, Scaled<I>, 12>().extract::<0>() == 0.0);
}

#[test]
fn bessel_in_tiers_are_monotone() {
    // The trip count IS the tier here, so a lower tier must be worse but never wrong. The
    // policy ladder requires monotonicity. This is what enforces it.
    for &(n, x, _, want_scaled) in BESSEL_IN.iter() {
        let v = V::splat(x);
        let best = rel(at_order_scaled!(n, v, Precision), want_scaled);
        let perf = rel(at_order_scaled!(n, v, Performance), want_scaled);
        assert!(best <= 6e-15, "Best e^-x I_{n}({x}) rel {best:e}");
        assert!(perf <= 5e-12, "Average e^-x I_{n}({x}) rel {perf:e}");
    }
}

#[test]
fn bessel_in_asymptotic_arm_is_reached_and_correct() {
    // Proof the large-x arm actually runs, not just that the numbers happen to be right:
    // the recurrence's trip count is `N + 24 + 0.35x`, so at x = 1e5 it would need ~35,000
    // serial divides and at x = 1e6 ~350,000. If this test returns promptly and accurately,
    // the series took over.
    let got = V::splat(100_000.0)
        .bessel_n_p::<Precision, Scaled<I>, 2>()
        .extract::<0>();
    assert!(
        rel(got, 0.0012615426067461744) <= 1e-14,
        "e^-x I_2(1e5): got {got}, want 0.0012615426067461744"
    );

    let got = V::splat(1e6).bessel_n_p::<Precision, Scaled<I>, 3>().extract::<0>();
    assert!(
        rel(got, 0.00039894053503190124) <= 1e-14,
        "e^-x I_3(1e6): got {got}, want 0.00039894053503190124"
    );
}

#[test]
fn bessel_in_asymptotic_seam_is_continuous() {
    // The handover joins two entirely unrelated algorithms (a continued fraction and a
    // divergent asymptotic series), so a step across the threshold is the failure this
    // catches, invisible to a coarse sweep. Order 2 crosses at x = 40, order 20 at
    // N^2/3 = 133. Both sides are checked against mpmath rather than against each other, so
    // agreeing on a wrong value does not pass.
    let below = V::splat(39.9).bessel_n_p::<Precision, Scaled<I>, 2>().extract::<0>();
    let above = V::splat(40.1).bessel_n_p::<Precision, Scaled<I>, 2>().extract::<0>();
    assert!(
        rel(below, 0.06022224749876019) <= 1e-14,
        "below the order-2 seam: {below}"
    );
    assert!(
        rel(above, 0.06008631752447881) <= 1e-14,
        "above the order-2 seam: {above}"
    );

    let below = V::splat(132.0).bessel_n_p::<Precision, Scaled<I>, 20>().extract::<0>();
    let above = V::splat(133.0).bessel_n_p::<Precision, Scaled<I>, 20>().extract::<0>();
    assert!(
        rel(below, 0.007616927724326666) <= 1e-14,
        "below the order-20 seam: {below}"
    );
    assert!(
        rel(above, 0.007675284513343219) <= 1e-14,
        "above the order-20 seam: {above}"
    );
}

/// Lanes on opposite arms inside one packet.
///
/// Needs a genuinely wide register: the `Vector<f64>` the rest of this file uses is the
/// 1-lane scalar seed, so a "mixed packet" written against it is a single lane and proves
/// nothing. Stamped per backend the way `elliptic.rs` does it.
macro_rules! mixed_packet_test {
    ($modname:ident, $backend:ty, $f64reg:ident) => {
        mod $modname {
            use super::*;

            type W = Vector<<$backend as Simd>::$f64reg>;

            #[test]
            fn bessel_in_mixed_packet_agrees_with_uniform_ones() {
                // The recurrence's trip count follows the widest lane that still needs it,
                // and the asymptotic lanes are pinned to a zero start so they cannot drag it.
                // A bug there shows up as a lane running the wrong number of iterations, which
                // a uniform packet can never expose.
                let mixed = W::splat(3.0).insert::<0>(500.0);
                let got = mixed.bessel_n_p::<Precision, Scaled<I>, 2>();

                let want_hi = W::splat(500.0)
                    .bessel_n_p::<Precision, Scaled<I>, 2>()
                    .extract::<0>();
                let want_lo = W::splat(3.0)
                    .bessel_n_p::<Precision, Scaled<I>, 2>()
                    .extract::<0>();

                assert!(
                    rel(got.extract::<0>(), want_hi) <= 1e-15,
                    "asymptotic lane moved when packed with recurrence lanes: {} vs {want_hi}",
                    got.extract::<0>()
                );
                for i in 1..W::LANES {
                    let v = got.extractv(i);
                    assert!(
                        rel(v, want_lo) <= 1e-15,
                        "recurrence lane {i} moved when packed with an asymptotic lane: {v} vs {want_lo}"
                    );
                }
            }
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod mixed_x86 {
    use super::*;
    use thermite::backend::x86_v2::X86V2;
    use thermite::backend::x86_v3::X86V3;
    mixed_packet_test!(v3, X86V3, f64x4);
    mixed_packet_test!(v2, X86V2, f64x2);
}

#[cfg(target_arch = "aarch64")]
mod mixed_neon {
    use super::*;
    use thermite::backend::neon::Neon;
    mixed_packet_test!(neon, Neon, f64x2);
}

include!("bessel_ref/table_k.rs");

macro_rules! k_at_order {
    ($n:expr, $v:expr, $tier:ty, $f:ty) => {
        match $n {
            0 => $v.bessel_n_p::<$tier, $f, 0>().extract::<0>(),
            1 => $v.bessel_n_p::<$tier, $f, 1>().extract::<0>(),
            2 => $v.bessel_n_p::<$tier, $f, 2>().extract::<0>(),
            3 => $v.bessel_n_p::<$tier, $f, 3>().extract::<0>(),
            5 => $v.bessel_n_p::<$tier, $f, 5>().extract::<0>(),
            8 => $v.bessel_n_p::<$tier, $f, 8>().extract::<0>(),
            12 => $v.bessel_n_p::<$tier, $f, 12>().extract::<0>(),
            20 => $v.bessel_n_p::<$tier, $f, 20>().extract::<0>(),
            _ => unreachable!("order {} not in the table's grid", $n),
        }
    };
}

#[test]
fn bessel_k_matches_mpmath_f64() {
    let mut worst_s = 0.0f64;
    let mut worst_u = 0.0f64;
    let mut bad = 0usize;
    for &(n, x, want, want_scaled) in BESSEL_K.iter() {
        let v = V::splat(x);

        let got_s = k_at_order!(n, v, Precision, Scaled<K>);
        // Flat, deliberately. The upward recurrence sums terms of like sign (no cancellation),
        // so despite running N steps its error does NOT grow with the order: measured
        // 1.42e-15 worst at order 20, the same as at order 0. A gate written `a + b*n` (the
        // first guess here) would have passed while testing nothing.
        let tol = 4e-15;
        let e = rel(got_s, want_scaled);
        if e > tol {
            bad += 1;
            std::println!("e^x K_{n}({x}): got {got_s}, want {want_scaled}, rel {e:e} (tol {tol:e})");
        }
        worst_s = worst_s.max(e);

        if want.is_finite() && want > 0.0 {
            let got_u = k_at_order!(n, v, Precision, K);
            let e = rel(got_u, want);
            if e > tol + 1e-16 * x {
                bad += 1;
                std::println!("K_{n}({x}): got {got_u}, want {want}, rel {e:e}");
            }
            worst_u = worst_u.max(e);
        }
    }
    std::println!(
        "K f64: worst scaled {worst_s:e}, worst unscaled {worst_u:e} over {} rows",
        BESSEL_K.len()
    );
    assert!(bad == 0, "{bad} rows over the gate");
}

#[test]
fn bessel_k_is_undefined_left_of_the_origin() {
    // K has a branch cut on the negative axis. No parity to exploit, unlike I. NaN rather
    // than a mirrored value, and the pole at zero is a real infinity.
    for &x in &[-1e-9f64, -0.5, -1.0, -3.0, -100.0] {
        for got in [
            V::splat(x).bessel_n_p::<Precision, K, 0>().extract::<0>(),
            V::splat(x).bessel_n_p::<Precision, K, 1>().extract::<0>(),
            V::splat(x).bessel_n_p::<Precision, K, 3>().extract::<0>(),
        ] {
            assert!(got.is_nan(), "K({x}) should be NaN, got {got}");
        }
    }
    assert!(
        V::splat(0.0).bessel_n_p::<Precision, K, 0>().extract::<0>() == f64::INFINITY,
        "K_0(0) should be +inf"
    );
}

#[test]
fn bessel_k_scaled_survives_where_unscaled_underflows() {
    // The mirror of the I story: there the unscaled form overflows, here it underflows to a
    // flat zero while the scaled one stays an ordinary number near sqrt(pi/2x).
    assert!(V::splat(800.0).bessel_n_p::<Precision, K, 0>().extract::<0>() == 0.0);
    let s = V::splat(800.0).bessel_n_p::<Precision, Scaled<K>, 0>().extract::<0>();
    let want = (core::f64::consts::PI / 1600.0).sqrt();
    assert!(
        rel(s, want) <= 1e-3 && s > 0.0,
        "e^x K_0(800): got {s}, expected near sqrt(pi/2x) = {want}"
    );
}

#[test]
fn bessel_k_wronskian() {
    // I_n(x) K_{n+1}(x) + I_{n+1}(x) K_n(x) = 1/x, exactly: the defining relation between
    // the two kinds. It crosses all four kernels plus both recurrences (I downward, K upward),
    // which no single-family test can do.
    for &x in &[0.1f64, 0.5, 1.0, 2.0, 5.0, 20.0] {
        let v = V::splat(x);
        // Scaled forms, so the e^x and e^-x cancel and nothing overflows.
        let pairs = [
            (
                v.bessel_n_p::<Precision, Scaled<I>, 0>().extract::<0>(),
                v.bessel_n_p::<Precision, Scaled<K>, 1>().extract::<0>(),
                v.bessel_n_p::<Precision, Scaled<I>, 1>().extract::<0>(),
                v.bessel_n_p::<Precision, Scaled<K>, 0>().extract::<0>(),
            ),
            (
                v.bessel_n_p::<Precision, Scaled<I>, 2>().extract::<0>(),
                v.bessel_n_p::<Precision, Scaled<K>, 3>().extract::<0>(),
                v.bessel_n_p::<Precision, Scaled<I>, 3>().extract::<0>(),
                v.bessel_n_p::<Precision, Scaled<K>, 2>().extract::<0>(),
            ),
        ];
        for (i, (ia, kb, ib, ka)) in pairs.iter().enumerate() {
            let got = ia * kb + ib * ka;
            assert!(
                rel(got, 1.0 / x) <= 1e-12,
                "Wronskian at x={x}, pair {i}: got {got}, want {}",
                1.0 / x
            );
        }
    }
}

#[test]
fn bessel_k_recurrence_joins_the_closed_forms() {
    // K_2 = K_0 + (2/x) K_1, from the closed forms alone.
    for &x in &[0.25f64, 0.9, 1.0, 1.1, 4.0, 30.0] {
        let v = V::splat(x);
        let want = v.bessel_n_p::<Precision, Scaled<K>, 0>().extract::<0>()
            + (2.0 / x) * v.bessel_n_p::<Precision, Scaled<K>, 1>().extract::<0>();
        let got = v.bessel_n_p::<Precision, Scaled<K>, 2>().extract::<0>();
        assert!(rel(got, want) <= 1e-14, "K_2({x}): recurrence {got}, identity {want}");
    }
}

#[test]
fn bessel_k_seam_is_continuous() {
    // x = 1 joins the log-singular series to the asymptotic rational.
    for &n in &[0u32, 1, 3] {
        let below = k_at_order!(n, V::splat(1.0f64.next_down()), Precision, Scaled<K>);
        let above = k_at_order!(n, V::splat(1.0f64.next_up()), Precision, Scaled<K>);
        assert!(
            rel(below, above) <= 1e-13,
            "K_{n} jumps across the x = 1 seam: {below} vs {above}"
        );
    }
}

include!("bessel_ref/table_jy.rs");

/// Envelope-relative error, the contract for a function with zeros.
///
/// `|got - want| / amplitude`, with amplitude `sqrt(2/(pi x))` above 1 and simply 1 below it
/// (where the functions are O(1) and not yet oscillating). A plain relative comparison is
/// meaningless at a zero (the true value is 0). This grid deliberately lands on the first
/// zero of each function, so a relative gate would be unpassable rather than merely loose.
fn env_rel(got: f64, want: f64, x: f64) -> f64 {
    // Scale by whichever is larger, the envelope or the value itself. Near the origin `Y` is
    // huge (Y_1(0.001) is -636) and dividing by the envelope alone would call a perfectly
    // good 1.8e-16 relative error a failure. Out at the zeros the value is ~0 and only the
    // envelope means anything. The max of the two is the contract in both regimes.
    let amp = (2.0 / (core::f64::consts::PI * x.max(1e-300))).sqrt();
    (got - want).abs() / amp.max(want.abs())
}

#[test]
fn bessel_jy_matches_mpmath_f64() {
    let mut worst = [0.0f64; 4];
    let mut bad = 0usize;
    for &(x, j0, j1, y0, y1) in BESSEL_JY.iter() {
        let v = V::splat(x);
        let got = [
            v.bessel_n_p::<Precision, J, 0>().extract::<0>(),
            v.bessel_n_p::<Precision, J, 1>().extract::<0>(),
            v.bessel_n_p::<Precision, Y, 0>().extract::<0>(),
            v.bessel_n_p::<Precision, Y, 1>().extract::<0>(),
        ];
        let want = [j0, j1, y0, y1];
        let names = ["J0", "J1", "Y0", "Y1"];
        for k in 0..4 {
            // Y diverges at the origin, so the generator clamped those entries to zero.
            if k >= 2 && x < 1e-6 {
                continue;
            }
            let e = env_rel(got[k], want[k], x);
            if e > 1e-13 {
                bad += 1;
                std::println!(
                    "{}({x}): got {}, want {}, envelope-rel {e:e}",
                    names[k],
                    got[k],
                    want[k]
                );
            }
            worst[k] = worst[k].max(e);
        }
    }
    std::println!(
        "J/Y f64 envelope-relative: J0 {:e}, J1 {:e}, Y0 {:e}, Y1 {:e} over {} rows",
        worst[0],
        worst[1],
        worst[2],
        worst[3],
        BESSEL_JY.len()
    );
    assert!(bad == 0, "{bad} entries over the gate");
}

#[test]
fn bessel_j_keeps_relative_accuracy_at_the_first_zeros() {
    // The stronger sub-8 guarantee, and the reason the fits are root-factored. Right beside a
    // zero the value is tiny while the polynomial coefficients are not, so an unfactored form
    // loses every digit here.
    for &(x, n) in &[(2.404825557695773_f64, 0u32), (3.831705970207512, 1)] {
        let got = if n == 0 {
            V::splat(x).bessel_n_p::<Precision, J, 0>().extract::<0>()
        } else {
            V::splat(x).bessel_n_p::<Precision, J, 1>().extract::<0>()
        };
        // The zero itself is not representable, so the true value at the nearest double is
        // around 1e-16. All that can be asked is that the result is that small, not ~1e-8.
        assert!(
            got.abs() < 1e-15,
            "J_{n} at its first zero {x}: got {got} - root factoring was lost"
        );
        // A step away it must track the derivative linearly, which an unfactored fit cannot.
        let h = 1e-7;
        let near = if n == 0 {
            V::splat(x + h).bessel_n_p::<Precision, J, 0>().extract::<0>()
        } else {
            V::splat(x + h).bessel_n_p::<Precision, J, 1>().extract::<0>()
        };
        let slope = near / h;
        assert!(slope.abs() > 0.1, "J_{n} flat beside its zero: slope {slope}");
    }
}

#[test]
fn bessel_y_is_undefined_left_of_the_origin() {
    for &x in &[-1e-9f64, -1.0, -5.0, -100.0] {
        for got in [
            V::splat(x).bessel_n_p::<Precision, Y, 0>().extract::<0>(),
            V::splat(x).bessel_n_p::<Precision, Y, 1>().extract::<0>(),
        ] {
            assert!(got.is_nan(), "Y({x}) should be NaN, got {got}");
        }
    }
    assert!(
        V::splat(0.0).bessel_n_p::<Precision, Y, 0>().extract::<0>() == f64::NEG_INFINITY,
        "Y_0(0) should be -inf"
    );
}

#[test]
fn bessel_j_parity_is_exact() {
    for &x in &[1e-8f64, 0.5, 3.0, 4.0, 7.9, 8.0, 8.1, 50.0] {
        let (p, m) = (V::splat(x), V::splat(-x));
        assert!(
            p.bessel_n_p::<Precision, J, 0>().extract::<0>().to_bits()
                == m.bessel_n_p::<Precision, J, 0>().extract::<0>().to_bits(),
            "J_0 not even at {x}"
        );
        let a = p.bessel_n_p::<Precision, J, 1>().extract::<0>();
        let b = m.bessel_n_p::<Precision, J, 1>().extract::<0>();
        assert!(a.to_bits() == (-b).to_bits(), "J_1 not odd at {x}: {a} vs {b}");
    }
}

#[test]
fn bessel_jy_wronskian() {
    // The Wronskian of the pair is 2/(pi x). At order 0 the derivatives are -J_1 and -Y_1, so
    // it reads J_1 Y_0 - J_0 Y_1 = 2/(pi x). One identity crossing all four kernels, and
    // holding AT the zeros, where nothing relative can be checked at all.
    for &x in &[0.5f64, 1.0, 2.404825557695773, 4.0, 7.9, 8.1, 20.0, 100.0] {
        let v = V::splat(x);
        let got = v.bessel_n_p::<Precision, J, 1>().extract::<0>() * v.bessel_n_p::<Precision, Y, 0>().extract::<0>()
            - v.bessel_n_p::<Precision, J, 0>().extract::<0>() * v.bessel_n_p::<Precision, Y, 1>().extract::<0>();
        let want = 2.0 / (core::f64::consts::PI * x);
        assert!(rel(got, want) <= 1e-13, "Wronskian at {x}: got {got}, want {want}");
    }
}

#[test]
fn bessel_jy_seams_are_continuous() {
    // 4 and 8 for J, 3 / 5.5 / 8 for Y_0, 4 / 8 for Y_1. Each joins two unrelated fits.
    for &x in &[3.0f64, 4.0, 5.5, 8.0] {
        let d = V::splat(x.next_down());
        let u = V::splat(x.next_up());
        for (name, lo, hi) in [
            (
                "J0",
                d.bessel_n_p::<Precision, J, 0>().extract::<0>(),
                u.bessel_n_p::<Precision, J, 0>().extract::<0>(),
            ),
            (
                "J1",
                d.bessel_n_p::<Precision, J, 1>().extract::<0>(),
                u.bessel_n_p::<Precision, J, 1>().extract::<0>(),
            ),
            (
                "Y0",
                d.bessel_n_p::<Precision, Y, 0>().extract::<0>(),
                u.bessel_n_p::<Precision, Y, 0>().extract::<0>(),
            ),
            (
                "Y1",
                d.bessel_n_p::<Precision, Y, 1>().extract::<0>(),
                u.bessel_n_p::<Precision, Y, 1>().extract::<0>(),
            ),
        ] {
            assert!(
                env_rel(lo, hi, x) <= 1e-13,
                "{name} jumps across the x = {x} seam: {lo} vs {hi}"
            );
        }
    }
}

#[test]
fn bessel_jy_agrees_with_libm() {
    // The first external scalar baseline in the whole Bessel arc. libm ships j0/j1/y0/y1 (the
    // C/POSIX XSI set) even though it has no modified Bessel at all, so this is the one
    // family where an independent implementation can be checked against. It catches a class
    // mpmath cannot: a shared misreading of the reference formulae.
    //
    // libm is itself only envelope-accurate and is the fdlibm four-region design this kernel
    // deliberately did not copy, so this is an agreement check, not a tighter oracle.
    for &(x, ..) in BESSEL_JY.iter() {
        if x <= 0.0 || x > 1e5 {
            continue;
        }
        let v = V::splat(x);
        for (name, got, want) in [
            ("j0", v.bessel_n_p::<Precision, J, 0>().extract::<0>(), libm::j0(x)),
            ("j1", v.bessel_n_p::<Precision, J, 1>().extract::<0>(), libm::j1(x)),
            ("y0", v.bessel_n_p::<Precision, Y, 0>().extract::<0>(), libm::y0(x)),
            ("y1", v.bessel_n_p::<Precision, Y, 1>().extract::<0>(), libm::y1(x)),
        ] {
            let e = env_rel(got, want, x);
            assert!(
                e <= 1e-11,
                "{name}({x}): thermite {got}, libm {want}, envelope-rel {e:e}"
            );
        }
    }
}

include!("bessel_ref/table_jyn.rs");

macro_rules! jyn_at {
    ($n:expr, $v:expr, $f:ty) => {
        match $n {
            2 => $v.bessel_n_p::<Precision, $f, 2>().extract::<0>(),
            3 => $v.bessel_n_p::<Precision, $f, 3>().extract::<0>(),
            5 => $v.bessel_n_p::<Precision, $f, 5>().extract::<0>(),
            8 => $v.bessel_n_p::<Precision, $f, 8>().extract::<0>(),
            12 => $v.bessel_n_p::<Precision, $f, 12>().extract::<0>(),
            20 => $v.bessel_n_p::<Precision, $f, 20>().extract::<0>(),
            30 => $v.bessel_n_p::<Precision, $f, 30>().extract::<0>(),
            _ => unreachable!("order {} not in the grid", $n),
        }
    };
}

#[test]
fn bessel_jn_higher_orders_match_mpmath() {
    // J_n runs two arms: forward when N < x (measured exact there), downward ratio otherwise.
    // The grid straddles that crossover at every order.
    let mut worst = 0.0f64;
    let mut bad = 0usize;
    for &(n, x, want, _) in BESSEL_JYN.iter() {
        let got = jyn_at!(n, V::splat(x), J);
        let e = env_rel(got, want, x);
        if e > 1e-13 {
            bad += 1;
            std::println!("J_{n}({x}): got {got}, want {want}, envelope-rel {e:e}");
        }
        worst = worst.max(e);
    }
    std::println!("J_n envelope-relative: worst {worst:e} over {} rows", BESSEL_JYN.len());
    assert!(bad == 0, "{bad} rows over the gate");
}

#[test]
fn bessel_yn_higher_orders_match_mpmath() {
    // Y_n is the dominant solution, so one upward arm covers everything. Looser gate than J:
    // the recurrence subtracts, so it accumulates a little cancellation over N steps:
    // measured 2.8e-14 in the model, against K's 1.4e-15 where the recurrence adds.
    let mut worst = 0.0f64;
    let mut bad = 0usize;
    for &(n, x, _, want) in BESSEL_JYN.iter() {
        let got = jyn_at!(n, V::splat(x), Y);
        let e = ((got - want) / want).abs();
        if e > 5e-13 {
            bad += 1;
            std::println!("Y_{n}({x}): got {got}, want {want}, rel {e:e}");
        }
        worst = worst.max(e);
    }
    std::println!("Y_n relative: worst {worst:e} over {} rows", BESSEL_JYN.len());
    assert!(bad == 0, "{bad} rows over the gate");
}

#[test]
fn bessel_jn_crossover_is_seamless() {
    // N < x picks forward, N >= x picks the downward ratio. The two arms share no arithmetic
    // at all, so a step across the crossover is the failure mode. It moves with N, which
    // a fixed-x sweep would never probe.
    for &n in &[3u32, 8, 20] {
        let x = n as f64;
        // ULP-adjacent, not `x +/- 1e-9`. At order 3 the derivative is about 0.177, so a 1e-9
        // step moves the value by 1.8e-10 all by itself. The first version of this test
        // compared two correct answers a billion ulps apart and called it a discontinuity.
        let below = jyn_at!(n, V::splat(x.next_down()), J);
        let above = jyn_at!(n, V::splat(x.next_up()), J);
        assert!(
            env_rel(below, above, x) <= 1e-13,
            "J_{n} jumps across its N = x crossover: {below} vs {above}"
        );
    }
}

#[test]
fn bessel_jn_survives_j0_zeros() {
    // The downward arm recovers J_N from J_0 times a product of ratios, which is exactly the
    // wrong thing to do where J_0 vanishes, at 2.404825..., 5.520078..., 8.653727... Boost
    // normalizes by J_0 unconditionally, but this kernel seeds from whichever of J_0/J_1 is
    // larger. These orders are >= x, so they take the downward arm.
    for &x in &[2.404825557695773_f64, 5.520078110286311, 8.653727912911013] {
        for &n in &[12u32, 20, 30] {
            let got = jyn_at!(n, V::splat(x), J);
            assert!(got.is_finite(), "J_{n}({x}) not finite at a zero of J_0: {got}");
        }
    }
    // And the values are right, not merely finite: order 12 at the first three J_0 zeros.
    for &(x, want) in &[
        (2.404825557695773_f64, 1.7053446163143602e-8_f64),
        (5.520078110286311, 0.0002241459696646508),
    ] {
        let got = V::splat(x).bessel_n_p::<Precision, J, 12>().extract::<0>();
        assert!(
            ((got - want) / want).abs() <= 1e-11,
            "J_12({x}): got {got}, want {want}"
        );
    }
}

#[test]
fn bessel_jyn_agrees_with_libm_at_higher_orders() {
    // libm has jn/yn as well as j0/j1/y0/y1, so the external baseline extends to every order.
    for &(n, x, ..) in BESSEL_JYN.iter() {
        let v = V::splat(x);
        let gj = jyn_at!(n, v, J);
        let gy = jyn_at!(n, v, Y);
        assert!(
            env_rel(gj, libm::jn(n as i32, x), x) <= 1e-10,
            "J_{n}({x}): thermite {gj}, libm {}",
            libm::jn(n as i32, x)
        );
        assert!(
            env_rel(gy, libm::yn(n as i32, x), x) <= 1e-10,
            "Y_{n}({x}): thermite {gy}, libm {}",
            libm::yn(n as i32, x)
        );
    }
}

#[test]
fn bessel_jyn_reference_tier_is_libm_exactly() {
    // The `Reference` contract: bit-identical to libm, lane by lane, at every order.
    for &(n, x, ..) in BESSEL_JYN.iter() {
        let v = V::splat(x);
        let gj = match n {
            2 => v
                .bessel_n_p::<thermite::math::policy::policies::Reference, J, 2>()
                .extract::<0>(),
            8 => v
                .bessel_n_p::<thermite::math::policy::policies::Reference, J, 8>()
                .extract::<0>(),
            20 => v
                .bessel_n_p::<thermite::math::policy::policies::Reference, J, 20>()
                .extract::<0>(),
            _ => continue,
        };
        assert!(
            gj.to_bits() == libm::jn(n as i32, x).to_bits(),
            "Reference J_{n}({x}) not bit-identical to libm: {gj} vs {}",
            libm::jn(n as i32, x)
        );
    }
}

/// The runtime-order forms, checked against the const-generic ones they must reproduce.
///
/// Deliberately NOT checked against mpmath here. The const forms already are, so the useful
/// question is whether the `v` variants agree with them. A shared reference table would let
/// both drift together, and the whole point of these is that they run the same recurrences
/// with the order in a register instead of in the type.
macro_rules! const_order {
    ($n:expr, $v:expr, $f:ty) => {
        match $n {
            0 => $v.bessel_n_p::<Precision, $f, 0>().extract::<0>(),
            1 => $v.bessel_n_p::<Precision, $f, 1>().extract::<0>(),
            2 => $v.bessel_n_p::<Precision, $f, 2>().extract::<0>(),
            3 => $v.bessel_n_p::<Precision, $f, 3>().extract::<0>(),
            5 => $v.bessel_n_p::<Precision, $f, 5>().extract::<0>(),
            8 => $v.bessel_n_p::<Precision, $f, 8>().extract::<0>(),
            12 => $v.bessel_n_p::<Precision, $f, 12>().extract::<0>(),
            _ => unreachable!(),
        }
    };
}

#[test]
fn bessel_runtime_order_matches_const_order() {
    use thermite::prelude::*;
    type S = <V as GenericVector>::Signed;

    for &n in &[0u32, 1, 2, 3, 5, 8, 12] {
        let nv = BesselOrder::Integer(S::splat(n as i64 as _));
        for &x in &[0.25f64, 0.9, 1.0, 2.0, 5.0, 7.75, 10.0, 20.0, 50.0] {
            let v = V::splat(x);

            // I and K: the scaled forms, so nothing leaves range at order 12.
            let got = v.bessel_p::<Precision, Scaled<I>>(nv).extract::<0>();
            let want = const_order!(n, v, Scaled<I>);
            assert!(
                rel(got, want) <= 1e-13,
                "I_{n}({x}) scaled: runtime {got}, const {want}"
            );

            let got = v.bessel_p::<Precision, Scaled<K>>(nv).extract::<0>();
            let want = const_order!(n, v, Scaled<K>);
            assert!(
                rel(got, want) <= 1e-13,
                "K_{n}({x}) scaled: runtime {got}, const {want}"
            );

            let got = v.bessel_p::<Precision, J>(nv).extract::<0>();
            let want = const_order!(n, v, J);
            assert!(
                env_rel(got, want, x) <= 1e-13,
                "J_{n}({x}): runtime {got}, const {want}"
            );

            let got = v.bessel_p::<Precision, Y>(nv).extract::<0>();
            let want = const_order!(n, v, Y);
            assert!(
                env_rel(got, want, x) <= 1e-12,
                "Y_{n}({x}): runtime {got}, const {want}"
            );
        }
    }
}

#[test]
fn bessel_runtime_order_handles_mixed_lanes() {
    // The point of the runtime form: different orders in one packet. A single-order packet
    // would pass even if the per-lane masking were broken, since every lane would freeze at the
    // same step.
    use thermite::backend::x86_v3::X86V3;
    type W = Vector<<X86V3 as Simd>::f64x4>;
    type WS = <W as GenericVector>::Signed;

    let x = W::splat(3.0);
    let orders = BesselOrder::Integer(WS::ZERO.insert::<0>(0).insert::<1>(1).insert::<2>(5).insert::<3>(8));
    let got = x.bessel_p::<Precision, J>(orders);

    let want = [
        V::splat(3.0).bessel_n_p::<Precision, J, 0>().extract::<0>(),
        V::splat(3.0).bessel_n_p::<Precision, J, 1>().extract::<0>(),
        V::splat(3.0).bessel_n_p::<Precision, J, 5>().extract::<0>(),
        V::splat(3.0).bessel_n_p::<Precision, J, 8>().extract::<0>(),
    ];
    for (i, w) in want.iter().enumerate() {
        let g = got.extractv(i);
        assert!(
            env_rel(g, *w, 3.0) <= 1e-13,
            "mixed-order lane {i} (order {}): runtime {g}, const {w}",
            [0, 1, 5, 8][i]
        );
    }

    // And with the arguments differing too, so lanes disagree about which arm to take: at
    // order 5, x = 2 goes downward and x = 20 goes forward.
    let xs = W::splat(2.0).insert::<2>(20.0).insert::<3>(20.0);
    let got = xs.bessel_p::<Precision, J>(BesselOrder::Integer(WS::splat(5)));
    for (i, xv) in [2.0f64, 2.0, 20.0, 20.0].iter().enumerate() {
        let w = V::splat(*xv).bessel_n_p::<Precision, J, 5>().extract::<0>();
        let g = got.extractv(i);
        assert!(
            env_rel(g, w, *xv) <= 1e-13,
            "mixed-arm lane {i} (x = {xv}): runtime {g}, const {w}"
        );
    }
}

#[test]
fn bessel_runtime_order_uses_the_asymptotic_arm() {
    // The runtime-order path must reach the same large-x arm the const path does. Without it
    // those lanes fall back to the recurrence, whose trip count IS the precision tier, which
    // measured 698 to 2634 ULP on the lower tiers over [-400, 400] before `asymptotic_series_v`
    // existed. `x` here is past `max(40, n^2/3)` for every order listed.
    use thermite::math::policy::policies::Performance;
    use thermite::prelude::*;
    type S = <V as GenericVector>::Signed;

    for &n in &[0u32, 1, 2, 5, 8] {
        let nv = BesselOrder::Integer(S::splat(n as i64 as _));
        for &x in &[60.0f64, 150.0, 400.0] {
            let v = V::splat(x);

            // Against the const form, at the DEFAULT tier, the one the arm was carrying.
            let got = v.bessel_p::<Performance, Scaled<I>>(nv).extract::<0>();
            let want = match n {
                0 => v.bessel_n_p::<Performance, Scaled<I>, 0>().extract::<0>(),
                1 => v.bessel_n_p::<Performance, Scaled<I>, 1>().extract::<0>(),
                2 => v.bessel_n_p::<Performance, Scaled<I>, 2>().extract::<0>(),
                5 => v.bessel_n_p::<Performance, Scaled<I>, 5>().extract::<0>(),
                _ => v.bessel_n_p::<Performance, Scaled<I>, 8>().extract::<0>(),
            };
            assert!(
                rel(got, want) <= 1e-13,
                "scaled I_{n}({x}) at Performance: runtime {got}, const {want}, rel {:e}",
                rel(got, want)
            );
        }
    }
}

/// `BesselOrder::simplify` narrows to the cheapest class the data actually needs.
///
/// The interesting case is the one that must NOT fire: `Real -> Thirds` would have to snap a
/// float to an unrepresentable rational, silently evaluating a different function.
#[test]
fn bessel_order_simplifies_to_the_cheapest_class() {
    use thermite::prelude::*;
    type S = <V as GenericVector>::Signed;

    // Names the variant, plus lane 0 of its payload, so the test can assert both without
    // needing PartialEq on a float vector.
    fn class(o: BesselOrder<V, S>) -> (&'static str, i64) {
        match o {
            BesselOrder::Integer(k) => ("integer", k.extract::<0>()),
            BesselOrder::HalfInteger(k) => ("half", k.extract::<0>()),
            BesselOrder::Thirds(k) => ("thirds", k.extract::<0>()),
            // Real carries no integer payload, so 0 is a placeholder the assertions ignore.
            BesselOrder::Real(_) => ("real", 0),
        }
    }

    let int = |k: i64| BesselOrder::<V, S>::Integer(S::splat(k));
    let half = |k: i64| BesselOrder::<V, S>::HalfInteger(S::splat(k));
    let third = |k: i64| BesselOrder::<V, S>::Thirds(S::splat(k));
    let real = |v: f64| BesselOrder::<V, S>::Real(V::splat(v));

    // Integer is already the floor.
    assert_eq!(class(int(3).simplify()), ("integer", 3));

    // k/2 collapses exactly when k is even, including negatives.
    assert_eq!(class(half(4).simplify()), ("integer", 2));
    assert_eq!(class(half(-6).simplify()), ("integer", -3));
    assert_eq!(class(half(3).simplify()).0, "half");

    // k/3 collapses exactly when 3 divides k.
    assert_eq!(class(third(6).simplify()), ("integer", 2));
    assert_eq!(class(third(-9).simplify()), ("integer", -3));
    assert_eq!(class(third(4).simplify()).0, "thirds");
    assert_eq!(class(third(1).simplify()).0, "thirds");

    // The headline case: a caller who reaches for Real and passes whole numbers gets the
    // fast path anyway. Halves too: 1/2 is exactly representable.
    assert_eq!(class(real(3.0).simplify()), ("integer", 3));
    assert_eq!(class(real(-2.0).simplify()), ("integer", -2));
    assert_eq!(class(real(2.5).simplify()), ("half", 5));
    assert_eq!(class(real(-1.5).simplify()), ("half", -3));

    // ...and the case that must NOT fire. fl(1/3) is not 1/3, so snapping it to Thirds(1)
    // would evaluate a different function. It stays Real, and so does a genuine irrational.
    assert_eq!(class(real(1.0 / 3.0).simplify()).0, "real");
    assert_eq!(class(real(2.0 / 3.0).simplify()).0, "real");
    assert_eq!(class(real(0.7).simplify()).0, "real");
}

/// Downgrading is per-packet: one lane that resists keeps the whole packet general, because a
/// SIMD packet runs one algorithm. A single-lane vector cannot show this, so it needs a real
/// register width.
#[test]
fn bessel_order_downgrade_needs_every_lane() {
    use thermite::backend::x86_v3::X86V3;
    use thermite::prelude::*;
    type W = Vector<<X86V3 as Simd>::f64x4>;
    type WS = <W as GenericVector>::Signed;

    let all_whole = BesselOrder::<W, WS>::Real(W::splat(2.0).insert::<1>(5.0).insert::<3>(-1.0));
    assert!(matches!(all_whole.simplify(), BesselOrder::Integer(_)));

    // One fractional lane, and the packet stays on the general path.
    let one_bad = BesselOrder::<W, WS>::Real(W::splat(2.0).insert::<2>(2.25));
    assert!(matches!(one_bad.simplify(), BesselOrder::Real(_)));

    // Same for the integer classes: one odd numerator holds the packet at half-integer.
    let one_odd = BesselOrder::<W, WS>::HalfInteger(WS::splat(4).insert::<3>(7));
    assert!(matches!(one_odd.simplify(), BesselOrder::HalfInteger(_)));
}

/// The user-facing payoff: asking for `Real` and passing whole numbers must give bit-identical
/// results to asking for `Integer`, not merely close ones. It is the same code path.
#[test]
fn bessel_real_order_downgrades_to_the_integer_path() {
    use thermite::prelude::*;
    type S = <V as GenericVector>::Signed;

    for &n in &[0i64, 1, 2, 5, 8] {
        for &x in &[0.5f64, 2.0, 7.75, 30.0] {
            let v = V::splat(x);
            let via_int = v.bessel_p::<Precision, J>(BesselOrder::Integer(S::splat(n)));
            let via_real = v.bessel_p::<Precision, J>(BesselOrder::Real(V::splat(n as f64)));
            assert_eq!(
                via_int.extract::<0>().to_bits(),
                via_real.extract::<0>().to_bits(),
                "J_{n}({x}): Integer and downgraded Real must be the same path"
            );
        }
    }
}

/// Negative integer orders, the reason the const parameter is `i32` rather than `usize`.
///
/// The four families reflect differently: `J_{-n} = (-1)^n J_n` and `Y_{-n} = (-1)^n Y_n`,
/// while `I_{-n} = I_n` and `K_{-n} = K_n` outright. A sign flip is exact, so these are
/// bit-for-bit assertions rather than tolerances. A tolerance here would pass even if the
/// reflection were computed by some second, slightly different route.
#[test]
fn bessel_negative_orders_reflect_exactly() {
    for &x in &[0.25f64, 0.9, 2.0, 5.0, 7.75, 12.0, 30.0] {
        let v = V::splat(x);

        // J: odd orders flip, even orders do not.
        assert_eq!(
            v.bessel_n_p::<Precision, J, -1>().extract::<0>().to_bits(),
            (-v.bessel_n_p::<Precision, J, 1>().extract::<0>()).to_bits(),
            "J_-1({x}) must be exactly -J_1"
        );
        assert_eq!(
            v.bessel_n_p::<Precision, J, -2>().extract::<0>().to_bits(),
            v.bessel_n_p::<Precision, J, 2>().extract::<0>().to_bits(),
            "J_-2({x}) must be exactly J_2"
        );
        assert_eq!(
            v.bessel_n_p::<Precision, J, -5>().extract::<0>().to_bits(),
            (-v.bessel_n_p::<Precision, J, 5>().extract::<0>()).to_bits(),
            "J_-5({x}) must be exactly -J_5"
        );

        // Y reflects the same way.
        assert_eq!(
            v.bessel_n_p::<Precision, Y, -1>().extract::<0>().to_bits(),
            (-v.bessel_n_p::<Precision, Y, 1>().extract::<0>()).to_bits(),
            "Y_-1({x}) must be exactly -Y_1"
        );
        assert_eq!(
            v.bessel_n_p::<Precision, Y, -4>().extract::<0>().to_bits(),
            v.bessel_n_p::<Precision, Y, 4>().extract::<0>().to_bits(),
            "Y_-4({x}) must be exactly Y_4"
        );

        // I and K are even in integer order: no flip at ANY order, odd included. This is the
        // half of the contract a copy-paste from J would silently break.
        for (got, want, name) in [
            (
                v.bessel_n_p::<Precision, Scaled<I>, -1>().extract::<0>(),
                v.bessel_n_p::<Precision, Scaled<I>, 1>().extract::<0>(),
                "I_-1",
            ),
            (
                v.bessel_n_p::<Precision, Scaled<I>, -3>().extract::<0>(),
                v.bessel_n_p::<Precision, Scaled<I>, 3>().extract::<0>(),
                "I_-3",
            ),
            (
                v.bessel_n_p::<Precision, Scaled<K>, -1>().extract::<0>(),
                v.bessel_n_p::<Precision, Scaled<K>, 1>().extract::<0>(),
                "K_-1",
            ),
            (
                v.bessel_n_p::<Precision, Scaled<K>, -3>().extract::<0>(),
                v.bessel_n_p::<Precision, Scaled<K>, 3>().extract::<0>(),
                "K_-3",
            ),
        ] {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{name}({x}) must equal the positive order"
            );
        }
    }
}

/// The `Reference` tier reflects too. It routes to libm, so this pins that the reflection is
/// applied to libm's value rather than quietly falling back to our own kernel.
#[test]
fn bessel_negative_orders_reflect_at_reference() {
    use thermite::math::policy::policies::Reference;

    for &x in &[0.5f64, 3.0, 9.0, 25.0] {
        let v = V::splat(x);
        assert_eq!(
            v.bessel_n_p::<Reference, J, -3>().extract::<0>().to_bits(),
            (-libm::jn(3, x)).to_bits(),
            "reference J_-3({x})"
        );
        assert_eq!(
            v.bessel_n_p::<Reference, Y, -2>().extract::<0>().to_bits(),
            libm::yn(2, x).to_bits(),
            "reference Y_-2({x})"
        );
    }
}

/// Negative orders through the runtime path, including a packet mixing signs. The signed
/// payload made these representable before it made them correct, so a mixed-sign packet is
/// the case that matters: the reflection is per-lane, not per-call.
#[test]
fn bessel_negative_runtime_orders_match_the_const_form() {
    use thermite::backend::x86_v3::X86V3;
    use thermite::prelude::*;
    type S = <V as GenericVector>::Signed;
    type W = Vector<<X86V3 as Simd>::f64x4>;
    type WS = <W as GenericVector>::Signed;

    for &x in &[0.75f64, 3.0, 6.5, 20.0] {
        let v = V::splat(x);
        let ord = |k: i64| BesselOrder::Integer(S::splat(k));

        assert_eq!(
            v.bessel_p::<Precision, J>(ord(-3)).extract::<0>().to_bits(),
            v.bessel_n_p::<Precision, J, -3>().extract::<0>().to_bits(),
            "runtime J_-3({x})"
        );
        assert_eq!(
            v.bessel_p::<Precision, Y>(ord(-2)).extract::<0>().to_bits(),
            v.bessel_n_p::<Precision, Y, -2>().extract::<0>().to_bits(),
            "runtime Y_-2({x})"
        );
        assert_eq!(
            v.bessel_p::<Precision, Scaled<I>>(ord(-2)).extract::<0>().to_bits(),
            v.bessel_n_p::<Precision, Scaled<I>, -2>().extract::<0>().to_bits(),
            "runtime I_-2({x})"
        );
        assert_eq!(
            v.bessel_p::<Precision, Scaled<K>>(ord(-3)).extract::<0>().to_bits(),
            v.bessel_n_p::<Precision, Scaled<K>, -3>().extract::<0>().to_bits(),
            "runtime K_-3({x})"
        );
    }

    // One packet, four orders, both signs and both parities.
    let x = W::splat(4.0);
    let orders = BesselOrder::Integer(WS::ZERO.insert::<0>(-3).insert::<1>(2).insert::<2>(-2).insert::<3>(5));
    let got = x.bessel_p::<Precision, J>(orders);
    let want = [
        V::splat(4.0).bessel_n_p::<Precision, J, -3>().extract::<0>(),
        V::splat(4.0).bessel_n_p::<Precision, J, 2>().extract::<0>(),
        V::splat(4.0).bessel_n_p::<Precision, J, -2>().extract::<0>(),
        V::splat(4.0).bessel_n_p::<Precision, J, 5>().extract::<0>(),
    ];
    for (i, w) in want.iter().enumerate() {
        let g = got.extractv(i);
        assert!(
            env_rel(g, *w, 4.0) <= 1e-13,
            "mixed-sign lane {i}: runtime {g}, const {w}"
        );
    }
}

/// `BesselOrder::Real` end to end through the public API, the first point any of the
/// fractional-order machinery is reachable from outside the crate.
///
/// References are mpmath at 50 digits. `x` spans all three regions of the internal select.
#[test]
fn bessel_jv_and_yv_accept_real_orders() {
    use thermite::prelude::*;

    let env_rel = |got: f64, want: f64, x: f64| {
        let env = (2.0 / (core::f64::consts::PI * x)).sqrt();
        (got - want).abs() / env.max(want.abs())
    };

    const ROWS: &[(f64, f64, f64, f64)] = &[
        (0.3333333333333333, 0.5, 0.672830829497946, -0.8406278260433777),
        (0.3333333333333333, 4.0, -0.355427373454576, 0.17941676634394849),
        (0.3333333333333333, 25.0, 0.020097162141383115, -0.1582974186494417),
        (2.25, 1.5, 0.17207040140276186, -1.0952365333165308),
        (2.25, 9.0, 0.06283886940664354, -0.2625685257863794),
        (-0.6666666666666666, 0.5, 0.7683441764822306, 0.9324008688393952),
        (-0.6666666666666666, 25.0, 0.1581810722120303, 0.021154005791818097),
        (5.5, 9.0, 0.08438779749107019, 0.2848318597461538),
        (-2.25, 4.0, 0.2104805636761565, 0.37655633348423506),
    ];

    for &(nu, x, wj, wy) in ROWS {
        let v = V::splat(x);
        let ord = BesselOrder::Real(V::splat(nu));

        let j = v.bessel_p::<Precision, J>(ord).extract::<0>();
        let y = v.bessel_p::<Precision, Y>(ord).extract::<0>();

        assert!(env_rel(j, wj, x) <= 4e-14, "J_{nu}({x}): got {j}, want {wj}");
        assert!(env_rel(y, wy, x) <= 4e-14, "Y_{nu}({x}): got {y}, want {wy}");
    }
}

/// `HalfInteger` and `Thirds` are correct through the same path, and `Real` carrying a whole
/// number still downgrades to the integer kernel bit-for-bit.
#[test]
fn every_bessel_order_variant_resolves() {
    use thermite::prelude::*;
    type S = <V as GenericVector>::Signed;

    let x = V::splat(7.0);

    // J_{1/2}(x) = sqrt(2/pi x) sin x, exactly.
    let half = x
        .bessel_p::<Precision, J>(BesselOrder::HalfInteger(S::splat(1)))
        .extract::<0>();
    let want = (2.0 / (core::f64::consts::PI * 7.0)).sqrt() * 7.0f64.sin();
    assert!(
        ((half - want) / want).abs() <= 1e-14,
        "J_1/2(7): got {half}, want {want}"
    );

    // Thirds(1) is nu = 1/3, the same value the Real path gives for fl(1/3).
    let thirds = x
        .bessel_p::<Precision, J>(BesselOrder::Thirds(S::splat(1)))
        .extract::<0>();
    let real = x
        .bessel_p::<Precision, J>(BesselOrder::Real(V::splat(1.0 / 3.0)))
        .extract::<0>();
    assert!(
        ((thirds - real) / real).abs() <= 1e-15,
        "Thirds(1) and Real(1/3) disagree: {thirds} vs {real}"
    );

    // A whole number arriving as `Real` must still reach the integer kernel, unchanged.
    let via_real = x
        .bessel_p::<Precision, J>(BesselOrder::Real(V::splat(3.0)))
        .extract::<0>();
    let via_int = x
        .bessel_p::<Precision, J>(BesselOrder::Integer(S::splat(3)))
        .extract::<0>();
    assert_eq!(
        via_real.to_bits(),
        via_int.to_bits(),
        "Real(3.0) must downgrade to the integer path"
    );
}
