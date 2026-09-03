//! The runtime-order forms (`expint(x, n)`, `phi(x, n)`, `sph_bessel::<J>(x, n)`, ...) against
//! their const-generic twins (`*_n::<N>`), bit for bit.
//!
//! Same contract as `thermite/tests/runtime_degree.rs`: the runtime forms are the same
//! arithmetic with the order as a value, and the precomputed per-order constants (expint's
//! reciprocals and thresholds, phi's term counts) are the same values the const form folds.
//! So equality of bits, not a tolerance. The two orthonormal functions are the one place a
//! per-step constant is computed rather than folded (a correctly rounded division and
//! square root either way), and are held to the same standard here. If that ever
//! fails it is a finding, not a tolerance to loosen.

#![cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]

use thermite::Vector;
use thermite::math::policy::policies::{Performance, Precision};
use thermite::prelude::*;
use thermite_special::bessel::{I, J, K, Scaled, Y};
use thermite_special::{RealPrimalMathWithPolicy, RealSpecialMathWithPolicy, SpecialMathWithPolicy};

type D = Vector<f64>;
type F = Vector<f32>;

macro_rules! const_form {
    ($v:expr, $m:ident, $p:ty, $n:expr $(, $arg:expr)*) => {
        match $n {
            0 => $v.$m::<$p, 0>($($arg),*),
            1 => $v.$m::<$p, 1>($($arg),*),
            2 => $v.$m::<$p, 2>($($arg),*),
            3 => $v.$m::<$p, 3>($($arg),*),
            4 => $v.$m::<$p, 4>($($arg),*),
            5 => $v.$m::<$p, 5>($($arg),*),
            6 => $v.$m::<$p, 6>($($arg),*),
            7 => $v.$m::<$p, 7>($($arg),*),
            8 => $v.$m::<$p, 8>($($arg),*),
            9 => $v.$m::<$p, 9>($($arg),*),
            10 => $v.$m::<$p, 10>($($arg),*),
            12 => $v.$m::<$p, 12>($($arg),*),
            16 => $v.$m::<$p, 16>($($arg),*),
            _ => unreachable!(),
        }
    };
}

const ORDERS: [u32; 13] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16];

fn same_bits(what: &str, n: u32, x: f64, got: f64, want: f64) {
    assert!(
        got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
        "{what}: n = {n}, x = {x}: runtime {got:e} != const {want:e}"
    );
}

fn same_bits_f32(what: &str, n: u32, x: f32, got: f32, want: f32) {
    assert!(
        got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
        "{what}: n = {n}, x = {x}: runtime {got:e} != const {want:e}"
    );
}

/// Log-spaced positives with the awkward points, plus their negatives for the families
/// defined there.
fn grid(lo: f64, hi: f64, negatives: bool) -> Vec<f64> {
    let mut xs = vec![0.0, -0.0, f64::INFINITY, f64::NAN, 1.0, 2.0, 0.5];
    if negatives {
        xs.extend([-1.0, -2.0, f64::NEG_INFINITY]);
    }
    let mut x = lo;
    while x <= hi {
        xs.push(x);
        if negatives {
            xs.push(-x);
        }
        x *= 1.9;
    }
    xs
}

macro_rules! check_pair {
    ($what:literal, $m_rt:ident, $m_n:ident, $grid:expr, $orders:expr $(, $arg:expr)*) => {
        for &n in $orders {
            for &x in &$grid {
                let d = D::splat(x);
                same_bits(concat!($what, "/Precision"), n, x, d.$m_rt::<Precision>($($arg,)* n).extract::<0>(), const_form!(d, $m_n, Precision, n $(, $arg)*).extract::<0>());
                same_bits(concat!($what, "/Performance"), n, x, d.$m_rt::<Performance>($($arg,)* n).extract::<0>(), const_form!(d, $m_n, Performance, n $(, $arg)*).extract::<0>());

                let xf = x as f32;
                let f = F::splat(xf);
                same_bits_f32(concat!($what, "/f32"), n, xf, f.$m_rt::<Precision>($($arg,)* n).extract::<0>(), const_form!(f, $m_n, Precision, n $(, $arg)*).extract::<0>());
            }
        }
    };
}

#[test]
fn expint_matches_the_const_form() {
    // Spans the series arm, the continued fraction, the recurrence and the handover to the
    // fraction above the (precomputed) threshold.
    check_pair!("expint", expint_p, expint_n_p, grid(1e-6, 800.0, false), &ORDERS);
}

#[test]
fn phi_matches_the_const_form() {
    check_pair!("phi", phi_p, phi_n_p, grid(1e-6, 800.0, true), &ORDERS);
}

#[test]
fn hermite_matches_the_const_form() {
    check_pair!("hermite", hermite_p, hermite_n_p, grid(1e-3, 30.0, true), &ORDERS);
}

#[test]
fn hermite_function_matches_the_const_form() {
    check_pair!(
        "hermite_function",
        hermite_function_p,
        hermite_function_n_p,
        grid(1e-3, 30.0, true),
        &ORDERS
    );
}

#[test]
fn laguerre_family_matches_the_const_form() {
    for &alpha in &[0.0, 0.5, 2.0, -0.5] {
        let a = D::splat(alpha);
        for &n in &ORDERS {
            for &x in &grid(1e-3, 60.0, false) {
                let d = D::splat(x);
                same_bits(
                    "laguerre",
                    n,
                    x,
                    d.laguerre_p::<Precision>(a, n).extract::<0>(),
                    const_form!(d, laguerre_n_p, Precision, n, a).extract::<0>(),
                );
                same_bits(
                    "laguerre_function",
                    n,
                    x,
                    d.laguerre_function_p::<Precision>(a, n).extract::<0>(),
                    const_form!(d, laguerre_function_n_p, Precision, n, a).extract::<0>(),
                );
            }
        }
    }
    for &alpha in &[0i32, 1, 3] {
        for &n in &ORDERS {
            for &x in &grid(1e-3, 60.0, false) {
                let d = D::splat(x);
                same_bits(
                    "laguerre_function_i",
                    n,
                    x,
                    d.laguerre_function_i_p::<Precision>(alpha, n).extract::<0>(),
                    const_form!(d, laguerre_function_i_n_p, Precision, n, alpha).extract::<0>(),
                );
            }
        }
    }
}

#[test]
fn algebraic_sigmoid_matches_the_const_form() {
    check_pair!(
        "algebraic_sigmoid",
        algebraic_sigmoid_p,
        algebraic_sigmoid_n_p,
        grid(1e-6, 1e6, true),
        &ORDERS
    );

    for &n in &ORDERS {
        for &x in &grid(1e-6, 1e6, true) {
            let d = D::splat(x);
            let (y, dy) = d.algebraic_sigmoid_d_p::<Precision>(n);
            let (yn, dyn_) = const_form!(d, algebraic_sigmoid_d_n_p, Precision, n);
            same_bits("algebraic_sigmoid_d", n, x, y.extract::<0>(), yn.extract::<0>());
            same_bits("algebraic_sigmoid_d'", n, x, dy.extract::<0>(), dyn_.extract::<0>());
        }
    }
}

/// The spherical family takes its family as a type parameter, so the ident-based
/// `check_pair!` cannot name it. This is the same check with the marker threaded through.
macro_rules! const_sph {
    ($v:expr, $f:ty, $p:ty, $n:expr) => {
        match $n {
            0 => $v.sph_bessel_n_p::<$p, $f, 0>(),
            1 => $v.sph_bessel_n_p::<$p, $f, 1>(),
            2 => $v.sph_bessel_n_p::<$p, $f, 2>(),
            3 => $v.sph_bessel_n_p::<$p, $f, 3>(),
            4 => $v.sph_bessel_n_p::<$p, $f, 4>(),
            5 => $v.sph_bessel_n_p::<$p, $f, 5>(),
            6 => $v.sph_bessel_n_p::<$p, $f, 6>(),
            7 => $v.sph_bessel_n_p::<$p, $f, 7>(),
            8 => $v.sph_bessel_n_p::<$p, $f, 8>(),
            9 => $v.sph_bessel_n_p::<$p, $f, 9>(),
            10 => $v.sph_bessel_n_p::<$p, $f, 10>(),
            12 => $v.sph_bessel_n_p::<$p, $f, 12>(),
            16 => $v.sph_bessel_n_p::<$p, $f, 16>(),
            _ => unreachable!(),
        }
    };
}

macro_rules! check_sph {
    ($what:literal, $f:ty, $grid:expr) => {
        for &n in &ORDERS {
            for &x in &$grid {
                let d = D::splat(x);
                same_bits(
                    concat!($what, "/Precision"),
                    n,
                    x,
                    d.sph_bessel_p::<Precision, $f>(n).extract::<0>(),
                    const_sph!(d, $f, Precision, n).extract::<0>(),
                );
                same_bits(
                    concat!($what, "/Performance"),
                    n,
                    x,
                    d.sph_bessel_p::<Performance, $f>(n).extract::<0>(),
                    const_sph!(d, $f, Performance, n).extract::<0>(),
                );

                let xf = x as f32;
                let f = F::splat(xf);
                same_bits_f32(
                    concat!($what, "/f32"),
                    n,
                    xf,
                    f.sph_bessel_p::<Precision, $f>(n).extract::<0>(),
                    const_sph!(f, $f, Precision, n).extract::<0>(),
                );
            }
        }
    };
}

#[test]
fn spherical_bessel_matches_the_const_form() {
    let xs = grid(1e-4, 700.0, true);
    check_sph!("sph_bessel_j", J, xs);
    check_sph!("sph_bessel_y", Y, xs);
    check_sph!("sph_bessel_i", I, xs);
    check_sph!("sph_bessel_i_scaled", Scaled<I>, xs);
    check_sph!("sph_bessel_k", K, xs);
    check_sph!("sph_bessel_k_scaled", Scaled<K>, xs);
}
