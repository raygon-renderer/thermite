//! Generalized Laguerre polynomials: `laguerre::<N>` and its per-lane sibling `laguerrev`.
//!
//! References are the same three-term recurrence run in exact rational arithmetic
//! (Python `fractions`) and rounded to f64 exactly once, so the tolerances below
//! measure this kernel rather than a floating-point oracle's own drift.
//!
//! Coverage is organized around the three things that can independently go wrong:
//! the recurrence coefficients (checked against the reference table and the closed
//! forms), the `alpha` handling (checked by an exact cross-alpha identity, which a
//! table of values at one alpha cannot see), and the per-lane freeze in `laguerrev`
//! (checked at a real vector width, since a one-lane vector exits the loop the
//! moment its own degree is reached and never freezes at all).
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]
#![allow(clippy::excessive_precision)]

use thermite::Vector;
use thermite::backend::scalar::Scalar;
use thermite::prelude::*;
use thermite_special::SpecialMath;

type D = Vector<f64>;
type F = Vector<f32>;

/// A real vector width, on the always-available scalar backend. Needed for
/// `laguerrev`: with one lane the "some lanes still running" path is unreachable.
type D4 = thermite::simd::f64x4<Scalar>;
type U4 = thermite::simd::u64x4<Scalar>;

#[track_caller]
fn close(name: &str, got: f64, want: f64, tol: f64) {
    // Mixed absolute/relative: several probes sit where the recurrence's terms cancel
    // down to a value far below their own magnitude, and a pure relative bound there
    // measures the conditioning of the problem rather than the kernel.
    let err = (got - want).abs() / want.abs().max(1.0);
    assert!(err <= tol, "{name}: got {got:?}, want {want:?} (err {err:e})");
}

fn lag<const N: usize>(x: f64, alpha: f64) -> f64 {
    D::splat(x).laguerre::<N>(D::splat(alpha)).extract::<0>()
}

// --- Values against the exact rational recurrence ---

macro_rules! check {
    ($($alpha:expr, $n:literal, $x:expr => $want:expr;)*) => {$(
        close(
            &format!("L_{}^({})({})", $n, $alpha, $x),
            lag::<$n>($x, $alpha),
            $want,
            1e-14,
        );
    )*};
}

#[test]
fn laguerre_matches_exact_rational_reference() {
    // alpha = 0: the ordinary Laguerre polynomials.
    check! {
        0.0, 0, 0.5 => 1.0;
        0.0, 1, 0.5 => 5.00000000000000000e-01;
        0.0, 1, 7.0 => -6.00000000000000000e+00;
        0.0, 2, 0.5 => 1.25000000000000000e-01;
        0.0, 2, 2.5 => -8.75000000000000000e-01;
        0.0, 2, 7.0 => 1.15000000000000000e+01;
        0.0, 3, 0.5 => -1.45833333333333343e-01;
        0.0, 3, 2.5 => 2.70833333333333315e-01;
        0.0, 3, 7.0 => -3.66666666666666652e+00;
        0.0, 5, 0.5 => -4.45572916666666652e-01;
        0.0, 5, 2.5 => 1.03255208333333326e+00;
        0.0, 5, 7.0 => -5.16666666666666718e-01;
        0.0, 8, 0.5 => -4.98362998356894848e-01;
        0.0, 8, 2.5 => -4.10568479507688489e-01;
        0.0, 8, 7.0 => 3.20659722222222221e-01;
        0.0, 12, 0.5 => -2.31649638865519147e-01;
        0.0, 12, 2.5 => -5.45844745179816959e-01;
        0.0, 12, 7.0 => 1.22284522014122699e+00;
    }

    // alpha = 2: a positive integer order, as the hydrogen radial part uses.
    check! {
        2.0, 1, 0.5 => 2.50000000000000000e+00;
        2.0, 1, 7.0 => -4.00000000000000000e+00;
        2.0, 2, 2.5 => -8.75000000000000000e-01;
        2.0, 3, 0.5 => 5.60416666666666696e+00;
        2.0, 3, 2.5 => -1.97916666666666674e+00;
        2.0, 5, 0.5 => 7.45546874999999964e+00;
        2.0, 5, 7.0 => -6.76666666666666661e+00;
        2.0, 8, 2.5 => 3.22624327644469266e+00;
        2.0, 8, 7.0 => 7.77482638888888911e+00;
        2.0, 12, 0.5 => -5.94609817864007906e-01;
        2.0, 12, 7.0 => -1.03486462571315005e+01;
    }

    // Half-integer alpha, both signs: the Boys/Kummer parameter family.
    check! {
        0.5, 1, 7.0 => -5.50000000000000000e+00;
        0.5, 2, 7.0 => 8.87500000000000000e+00;
        0.5, 3, 2.5 => -4.16666666666666685e-01;
        0.5, 5, 0.5 => -2.43749999999999994e-01;
        0.5, 5, 7.0 => -4.00494791666666661e+00;
        0.5, 8, 7.0 => 4.15255466037326393e+00;
        0.5, 12, 2.5 => -1.23496311210781973e+00;
        -0.5, 1, 0.5 => 0.0;
        -0.5, 2, 2.5 => -2.50000000000000000e-01;
        -0.5, 3, 7.0 => -8.72916666666666607e+00;
        -0.5, 5, 2.5 => 6.04166666666666630e-01;
        -0.5, 8, 7.0 => -3.37167494032118054e+00;
        -0.5, 12, 2.5 => -1.20924121456483652e-02;
        -0.5, 12, 7.0 => 3.95179831741724019e+00;
    }
}

#[test]
fn laguerre_closed_forms() {
    for &alpha in &[0.0, 2.0, 0.5, -0.5, 3.25] {
        for &x in &[0.0, 0.5, 2.5, 7.0, -1.5] {
            close("L_0", lag::<0>(x, alpha), 1.0, 0.0);
            close("L_1", lag::<1>(x, alpha), 1.0 + alpha - x, 1e-15);

            // L_2^a(x) = x^2/2 - (a + 2) x + (a + 1)(a + 2)/2
            let want = 0.5 * x * x - (alpha + 2.0) * x + 0.5 * (alpha + 1.0) * (alpha + 2.0);
            close("L_2", lag::<2>(x, alpha), want, 1e-15);
        }
    }
}

#[test]
fn laguerre_at_zero_is_a_binomial_coefficient() {
    // L_n^a(0) = C(n + a, n), which is 1 for every n when a = 0.
    for_each_degree(|n, got| {
        if n <= 12 {
            close(&format!("L_{n}^(0)(0)"), got, 1.0, 0.0);
        }
    });

    close("L_3^(2)(0)", lag::<3>(0.0, 2.0), 10.0, 1e-15); // C(5,3)
    close("L_5^(2)(0)", lag::<5>(0.0, 2.0), 21.0, 1e-15); // C(7,5)
    close("L_4^(3)(0)", lag::<4>(0.0, 3.0), 35.0, 1e-15); // C(7,4)
}

/// Calls back with `(n, L_n^0(0))` for a fixed ladder of degrees, since `N` is a
/// const generic and cannot come from a loop variable.
fn for_each_degree(mut f: impl FnMut(usize, f64)) {
    f(0, lag::<0>(0.0, 0.0));
    f(1, lag::<1>(0.0, 0.0));
    f(2, lag::<2>(0.0, 0.0));
    f(3, lag::<3>(0.0, 0.0));
    f(5, lag::<5>(0.0, 0.0));
    f(8, lag::<8>(0.0, 0.0));
    f(12, lag::<12>(0.0, 0.0));
}

// --- The alpha axis, via an identity a single-alpha table cannot check ---

#[test]
fn laguerre_satisfies_the_cross_alpha_derivative_identity() {
    // Combining d/dx L_n^a = -L_{n-1}^{a+1} with x L_n^a' = n L_n^a - (n+a) L_{n-1}^a
    // gives an identity that is exact in rational arithmetic and ties three different
    // (n, alpha) instantiations together:
    //
    //     -x L_{n-1}^{a+1}(x) = n L_n^a(x) - (n + a) L_{n-1}^a(x)
    //
    // A wrong alpha offset anywhere in the recurrence breaks it, where a table of values at
    // one alpha would not.
    macro_rules! identity {
        ($n:literal, $($alpha:expr),+ $(,)?) => {$(
            for &x in &[0.5, 2.5, 7.0, 13.0 / 3.0] {
                let n = $n as f64;
                let lhs = -x * lag::<{ $n - 1 }>(x, $alpha + 1.0);
                let rhs = n * lag::<$n>(x, $alpha) - (n + $alpha) * lag::<{ $n - 1 }>(x, $alpha);
                close(&format!("identity n={} a={} x={x}", $n, $alpha), lhs, rhs, 1e-13);
            }
        )+};
    }

    identity!(1, 0.0, 2.0, 0.5, -0.5, 0.75);
    identity!(2, 0.0, 2.0, 0.5, -0.5, 0.75);
    identity!(3, 0.0, 2.0, 0.5, -0.5, 0.75);
    identity!(5, 0.0, 2.0, 0.5, -0.5, 0.75);
    identity!(8, 0.0, 2.0, 0.5, -0.5, 0.75);
}

// --- Per-lane degrees ---

#[test]
fn laguerrev_matches_laguerre_lane_by_lane() {
    // Degrees deliberately out of order and mixed with zero, so lanes retire at
    // different iterations and the loop keeps running after the first one is done.
    // A kernel that carries the previous term forward on a retired lane returns
    // L_{n-1} there and fails only in this arrangement.
    let orders: [[u64; 4]; 4] = [[0, 1, 2, 3], [3, 1, 5, 0], [8, 0, 3, 1], [12, 5, 8, 2]];

    let xs = [0.5, 2.5, 7.0, -1.5];
    let alphas = [0.0, 2.0, 0.5, -0.5];

    for ns in orders {
        for &x in &xs {
            for &alpha in &alphas {
                let got = D4::splat(x).laguerrev(D4::splat(alpha), U4::new(ns));

                for lane in 0..4 {
                    let want = match ns[lane] {
                        0 => lag::<0>(x, alpha),
                        1 => lag::<1>(x, alpha),
                        2 => lag::<2>(x, alpha),
                        3 => lag::<3>(x, alpha),
                        5 => lag::<5>(x, alpha),
                        8 => lag::<8>(x, alpha),
                        12 => lag::<12>(x, alpha),
                        n => unreachable!("degree {n} has no const-generic counterpart here"),
                    };

                    close(
                        &format!("laguerrev n={:?} lane={lane} x={x} a={alpha}", ns),
                        got.as_slice()[lane],
                        want,
                        1e-14,
                    );
                }
            }
        }
    }
}

#[test]
fn hermitev_matches_hermite_lane_by_lane() {
    // The sibling kernel, in the same arrangement and for the same reason. `laguerrev`
    // was written against `hermitev`, so if the per-lane freeze is wrong in one it is
    // worth knowing whether it is wrong in both.
    let orders: [[u64; 4]; 4] = [[0, 1, 2, 3], [3, 1, 5, 0], [8, 0, 3, 1], [10, 5, 8, 2]];

    for ns in orders {
        for &x in &[0.5, 2.5, -1.5] {
            let got = D4::splat(x).hermitev(U4::new(ns));

            for lane in 0..4 {
                let want = match ns[lane] {
                    0 => D::splat(x).hermite::<0>(),
                    1 => D::splat(x).hermite::<1>(),
                    2 => D::splat(x).hermite::<2>(),
                    3 => D::splat(x).hermite::<3>(),
                    5 => D::splat(x).hermite::<5>(),
                    8 => D::splat(x).hermite::<8>(),
                    10 => D::splat(x).hermite::<10>(),
                    n => unreachable!("degree {n} has no const-generic counterpart here"),
                }
                .extract::<0>();

                close(
                    &format!("hermitev n={:?} lane={lane} x={x}", ns),
                    got.as_slice()[lane],
                    want,
                    1e-13,
                );
            }
        }
    }
}

// --- f32 ---

#[test]
fn laguerre_f32_tracks_the_reference() {
    // (alpha, x, L_5^alpha(x)), same reference values as the f64 table.
    let probes: [(f32, f32, f32); 6] = [
        (0.0, 0.5, -4.45572916666666652e-01),
        (0.0, 2.5, 1.03255208333333326e+00),
        (2.0, 0.5, 7.45546874999999964e+00),
        (2.0, 7.0, -6.76666666666666661e+00),
        (0.5, 0.5, -2.43749999999999994e-01),
        (-0.5, 2.5, 6.04166666666666630e-01),
    ];

    for (alpha, x, want) in probes {
        let got = F::splat(x).laguerre::<5>(F::splat(alpha)).extract::<0>();
        close(&format!("f32 L_5^({alpha})({x})"), got as f64, want as f64, 1e-6);
    }
}

// --- Composite inheritance ---

#[test]
fn laguerre_differentiates_through_dual() {
    use thermite_dual::Dual;

    // `laguerre` is a default method written in vector arithmetic, so `Dual` gets it
    // with no implementation of its own. What that buys is the derivative for free,
    // and the exact identity d/dx L_n^a(x) = -L_{n-1}^{a+1}(x) says whether it is right.
    type DD = Dual<D, 1>;

    macro_rules! deriv {
        ($n:literal, $($alpha:expr),+ $(,)?) => {$(
            for &x in &[0.5, 2.5, 7.0] {
                let d = DD::variable(D::splat(x), 0)
                    .laguerre::<$n>(DD::constant(D::splat($alpha)));

                let value = d.value().extract::<0>();
                let slope = d.gradient()[0].extract::<0>();

                close(&format!("dual value n={} a={} x={x}", $n, $alpha), value, lag::<$n>(x, $alpha), 1e-14);
                close(
                    &format!("dual slope n={} a={} x={x}", $n, $alpha),
                    slope,
                    -lag::<{ $n - 1 }>(x, $alpha + 1.0),
                    1e-13,
                );
            }
        )+};
    }

    deriv!(1, 0.0, 2.0, 0.5);
    deriv!(3, 0.0, 2.0, 0.5);
    deriv!(5, 0.0, 2.0, -0.5);
    deriv!(8, 0.0, 2.0, 0.5);
}
