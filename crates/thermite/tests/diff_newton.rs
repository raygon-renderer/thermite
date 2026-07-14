//! Thorough coverage of `newtons_method` (`math/algorithms/mod.rs`).
//!
//! Newton's method here is a Newton/bisection hybrid with per-lane convergence
//! tracking, optional bracketing bounds, monotonicity-independent shrink-wrap,
//! a zero-derivative bisection fallback, and an optional `check_overflow` guard.
//! Each of those branches gets a dedicated scenario, on Scalar + V2 + V3 for
//! both `f32x4` and `f64x4`. Roots are known in closed form and checked against
//! `f64` oracles.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "wasm32",
    all(feature = "neon", target_arch = "aarch64")
))]

use thermite::Vector;
use thermite::math::algorithms::{newtons_method, prod_f, sum_f};
use thermite::math::policy::policies::{CheckOverflow, MaxIterations, Performance, Precision, UseCompensation};
use thermite::prelude::*;
use thermite::simd::Simd;

use thermite::backend::scalar::Scalar;

/// All scenarios for one 4-lane float vector type.
///
/// `$btol`/`$brel`: tolerance + root rel-error for the *bounded* cases (the
/// bracket collapses to ~tolerance, so the root is tight). `$utol`/`$urel`: the
/// looser function-space pair for the *unbounded* cases.
macro_rules! newton_scenarios {
    ($V:ty, $e:ty, $L:expr, $btol:expr, $brel:expr, $utol:expr, $urel:expr) => {{
        type V = $V;
        let sp = |x: f64| V::splat(x as $e);
        let rd = |v: V| -> [f64; 4] {
            let g = v.into_array();
            [g[0] as f64, g[1] as f64, g[2] as f64, g[3] as f64]
        };
        // per-lane relative check against f64 roots
        let chk = |name: &str, root: V, want: [f64; 4], rel: f64| {
            let g = rd(root);
            for i in 0..4 {
                assert!(
                    (g[i] - want[i]).abs() <= rel * want[i].abs().max(1.0),
                    "{} [{}] lane {}: got {} want {} (rel {})",
                    $L, name, i, g[i], want[i], rel
                );
            }
        };

        let a = V::new([2.0 as $e, 3.0 as $e, 5.0 as $e, 7.0 as $e]);
        let sqrt_a = [2f64.sqrt(), 3f64.sqrt(), 5f64.sqrt(), 7f64.sqrt()];
        let two = sp(2.0);
        let three = sp(3.0);
        let four = sp(4.0);

        // f(x) = x^2 - a (increasing for x>0), f'(x) = 2x
        let sq = |x: V| (x * x - a, x + x);
        // f(x) = a - x^2 (decreasing for x>0), f'(x) = -2x
        let dsq = |x: V| (a - x * x, -(x + x));

        // --- 1. Unbounded Newton, per-lane different roots -------------------
        {
            let (root, conv) = newtons_method::<V, Precision, _>(a, sp($utol), None, sq);
            assert!(conv.all(), "{} unbounded_sqrt did not converge", $L);
            chk("unbounded_sqrt", root, sqrt_a, $urel);
        }

        // --- 2. Bounded (increasing); bracket collapses to ~tolerance --------
        {
            let bounds = Some((V::ZERO, a + V::ONE));
            let (root, conv) = newtons_method::<V, Precision, _>(sp(0.1), sp($btol), bounds, sq);
            assert!(conv.all(), "{} bounded_increasing did not converge", $L);
            chk("bounded_increasing", root, sqrt_a, $brel);
        }

        // --- 3. Initial guess outside the bracket must be clamped in ---------
        {
            let bounds = Some((V::ZERO, a + V::ONE));
            let (root, conv) = newtons_method::<V, Precision, _>(sp(-1.0e3), sp($btol), bounds, sq);
            assert!(conv.all(), "{} clamp_outside did not converge", $L);
            chk("clamp_outside", root, sqrt_a, $brel);
        }

        // --- 4. Bounded *decreasing* function (min_is_negative = false) ------
        {
            let bounds = Some((V::ZERO, a + V::ONE));
            let (root, conv) = newtons_method::<V, Precision, _>(sp(0.1), sp($btol), bounds, dsq);
            assert!(conv.all(), "{} bounded_decreasing did not converge", $L);
            chk("bounded_decreasing", root, sqrt_a, $brel);
        }

        // --- 5. Zero derivative *at* the root: f(x)=x^3, root 0 --------------
        // f'(0)=0, so pure Newton stalls; the bisection fallback must still land on 0.
        {
            let cube = |x: V| (x * x * x, three * x * x);
            let bounds = Some((sp(-1.0), two));
            let (root, conv) = newtons_method::<V, Precision, _>(sp(0.5), sp($btol), bounds, cube);
            assert!(conv.all(), "{} zero_deriv_at_root did not converge", $L);
            // f(x)=x^3 so the function-space tolerance only pins |x| <= tol^(1/3)
            // (e.g. ~0.009 for the f32 tol of 1e-6). What matters is that the
            // bisection fallback landed *near* 0 rather than stalling/diverging.
            for (i, &g) in rd(root).iter().enumerate() {
                assert!(g.abs() <= 0.05, "{} zero_deriv_at_root lane {}: got {} want ~0", $L, i, g);
            }
        }

        // --- 6. Interior zero derivative: f(x)=x^3-2x^2+x-2, root 2 ----------
        // f'(x)=3x^2-4x+1 is zero at x=1/3 and x=1, both inside (0,3); Newton
        // steps that hit a near-zero derivative must defer to bisection.
        {
            let cubic = |x: V| {
                let x2 = x * x;
                (x2 * x - two * x2 + x - two, three * x2 - four * x + V::ONE)
            };
            let bounds = Some((V::ZERO, three));
            let (root, conv) = newtons_method::<V, Precision, _>(sp(0.5), sp($btol), bounds, cubic);
            assert!(conv.all(), "{} interior_zero_deriv did not converge", $L);
            chk("interior_zero_deriv", root, [2.0, 2.0, 2.0, 2.0], ($brel as f64).max(1e-3));
        }

        // --- 7. Bracket collapse with tolerance = 0 (must terminate) --------
        // No function-space or width tolerance is reachable; the loop must still
        // converge once the midpoint can no longer land strictly inside (FP ULP).
        {
            let f = |x: V| (x * x - two, x + x);
            let bounds = Some((V::ONE, two));
            let (root, conv) = newtons_method::<V, Precision, _>(sp(1.5), V::ZERO, bounds, f);
            assert!(conv.all(), "{} bracket_collapse did not terminate-as-converged", $L);
            chk("bracket_collapse", root, [2f64.sqrt(); 4], ($brel as f64).max(1e-3));
        }

        // --- 8. Large tolerance => converged immediately at the start --------
        {
            let f = |x: V| (x * x - two, x + x);
            let (root, conv) = newtons_method::<V, Precision, _>(sp(3.0), sp(1.0e9), None, f);
            assert!(conv.all(), "{} large_tol not all-converged", $L);
            chk("large_tol_immediate", root, [3.0, 3.0, 3.0, 3.0], 0.0); // unchanged start
        }

        // --- 9. Per-lane partial convergence is reported precisely -----------
        // One lane has a huge tolerance (converges on the first check), the rest
        // do not, under a 1-iteration cap.
        {
            let f = |x: V| (x * x - two, x + x);
            let tol = V::new([1.0e9 as $e, 1.0e-12 as $e, 1.0e-12 as $e, 1.0e-12 as $e]);
            let (_root, conv) = newtons_method::<V, MaxIterations<Performance, 1>, _>(sp(3.0), tol, None, f);
            let bits = rd(conv.select(V::ONE, V::ZERO));
            assert_eq!(bits, [1.0, 0.0, 0.0, 0.0], "{} per_lane_partial mask wrong", $L);
        }

        // --- 10. Non-convergence under a capped iteration count --------------
        {
            let f = |x: V| (x * x - two, x + x);
            let (_root, conv) =
                newtons_method::<V, MaxIterations<Performance, 1>, _>(sp(10.0), sp(1.0e-9), None, f);
            assert!(!conv.all(), "{} should NOT fully converge in 1 iteration", $L);
        }

        // --- 11. check_overflow = true AND false both converge (both arms) ---
        {
            let bounds = Some((V::ZERO, a + V::ONE));
            let (r_on, c_on) =
                newtons_method::<V, CheckOverflow<Precision, true>, _>(sp(0.1), sp($btol), bounds, sq);
            let (r_off, c_off) =
                newtons_method::<V, CheckOverflow<Precision, false>, _>(sp(0.1), sp($btol), bounds, sq);
            assert!(c_on.all() && c_off.all(), "{} check_overflow arms did not converge", $L);
            chk("overflow_on", r_on, sqrt_a, $brel);
            chk("overflow_off", r_off, sqrt_a, $brel);
        }
    }};
}

macro_rules! newton_suite {
    ($mod:ident, $backend:ty) => {
        mod $mod {
            use super::*;

            #[test]
            fn f32() {
                newton_scenarios!(
                    Vector<<$backend as Simd>::f32x4>,
                    f32,
                    concat!(stringify!($mod), " f32x4"),
                    1.0e-6,
                    2.0e-4,
                    2.0e-3,
                    3.0e-3
                );
            }

            #[test]
            fn f64() {
                newton_scenarios!(
                    Vector<<$backend as Simd>::f64x4>,
                    f64,
                    concat!(stringify!($mod), " f64x4"),
                    1.0e-12,
                    1.0e-9,
                    1.0e-10,
                    1.0e-7
                );
            }
        }
    };
}

// scalar is the always-available oracle (runs on every target).
newton_suite!(scalar, Scalar);

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;
    use thermite::backend::x86_v1::X86V1;
    use thermite::backend::x86_v2::X86V2;
    use thermite::backend::x86_v3::X86V3;
    newton_suite!(v3, X86V3);
    newton_suite!(v2, X86V2);
    newton_suite!(v1, X86V1);
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    use super::*;
    use thermite::backend::wasm::Wasm;
    newton_suite!(wasm, Wasm);
}

#[cfg(all(feature = "neon", target_arch = "aarch64"))]
mod neon {
    use super::*;
    use thermite::backend::neon::Neon;
    newton_suite!(neon, Neon);
}

/// `sum_f` / `prod_f` (`math/algorithms/mod.rs`). The compensated (`UseCompensation
/// <_, true>`) variant exercises the Kahan path, which is the *only* place
/// `GenericMask::swap` is used in numeric code — the un-compensated tests never
/// reach it (`sum`/`prod` start at 0/1, so the first term always triggers a swap).
#[test]
fn series_sum_and_prod() {
    // `sum_f`/`prod_f` are backend-agnostic; scalar `f64x4` (a real 4-lane
    // ArrayRegister) exercises the same convergence/Kahan/swap paths and runs
    // on every target.
    type V = Vector<<Scalar as Simd>::f64x4>;
    let tol = V::splat(1e-15);

    // geometric series sum_{n>=0} 0.5^n = 2, converged (terms fall below tol).
    let geom = |n: i64| V::splat(0.5f64.powi(n as i32));
    let plain = sum_f::<V, Performance, _>(tol, 0, 200, geom);
    let kahan = sum_f::<V, UseCompensation<Performance, true>, _>(tol, 0, 200, geom); // drives swap
    for r in [plain, kahan] {
        let v = r.expect("geometric sum should converge").into_array()[0];
        assert!((v - 2.0).abs() < 1e-9, "sum_f geometric = {v}");
    }

    // non-convergence -> Err(partial). f(n)=1 never falls below tol; sums 10 terms.
    let err = sum_f::<V, Performance, _>(V::splat(1e-12), 0, 10, |_| V::ONE);
    let p = err.expect_err("constant series must not converge").into_array()[0];
    assert!((p - 10.0).abs() < 1e-9, "partial sum = {p}");

    // prod_f: product of (1 + 0.5^(n+1)) converges (delta shrinks below tol).
    let pr =
        prod_f::<V, UseCompensation<Performance, true>, _>(tol, 0, 80, |n| V::splat(1.0 + 0.5f64.powi(n as i32 + 1)));
    let pv = pr.expect("converging product").into_array()[0];
    assert!(pv > 2.0 && pv < 3.0, "prod_f = {pv}");

    // prod_f non-convergence: f(n)=2 keeps doubling -> delta never small -> Err.
    let perr = prod_f::<V, Performance, _>(tol, 0, 8, |_| V::TWO);
    assert!(perr.is_err(), "doubling product must not converge");
}
