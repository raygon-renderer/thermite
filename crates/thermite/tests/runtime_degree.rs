//! The runtime-degree forms (`nth_root(x, n)`, `log_n(x, n)`, the `smoothstep` family) against
//! their const-generic twins (`*_n::<N>`), bit for bit.
//!
//! The runtime forms exist so a caller with a runtime degree (the Python bridge above all)
//! does not need a ladder of const instantiations. Their contract is that they compute the
//! same thing: same arithmetic, same table entries, same special cases, so the check is
//! equality of bits, not a tolerance, at every degree the ladder used to cover and beyond.

#![cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]

use thermite::Vector;
use thermite::math::policy::policies::{Performance, Precision};
use thermite::math::{RealMathWithPolicy, TranscendentalMathWithPolicy};
use thermite::prelude::*;

type D = Vector<f64>;
type F = Vector<f32>;

/// Calls the const form at a runtime `n`, one rung per degree the test sweeps.
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
            20 => $v.$m::<$p, 20>($($arg),*),
            _ => unreachable!(),
        }
    };
}

const DEGREES: [u32; 14] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20];

fn same_bits_f64(what: &str, n: u32, x: f64, got: f64, want: f64) {
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

fn roots_grid() -> Vec<f64> {
    let mut xs = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.0,
        -2.0,
        1e-300,
        1e300,
        0.5,
        f64::INFINITY,
        f64::NAN,
        -8.0,
        27.0,
    ];
    let mut x = 1e-6;
    while x < 1e6 {
        xs.push(x);
        xs.push(-x);
        x *= 3.7;
    }
    xs
}

#[test]
fn nth_root_matches_the_const_form() {
    for &n in &DEGREES {
        for &x in &roots_grid() {
            let d = D::splat(x);
            same_bits_f64(
                "nth_root/Precision",
                n,
                x,
                d.nth_root_p::<Precision>(n).extract::<0>(),
                const_form!(d, nth_root_n_p, Precision, n).extract::<0>(),
            );
            same_bits_f64(
                "nth_root/Performance",
                n,
                x,
                d.nth_root_p::<Performance>(n).extract::<0>(),
                const_form!(d, nth_root_n_p, Performance, n).extract::<0>(),
            );

            let xf = x as f32;
            let f = F::splat(xf);
            same_bits_f32(
                "nth_root/f32",
                n,
                xf,
                f.nth_root_p::<Precision>(n).extract::<0>(),
                const_form!(f, nth_root_n_p, Precision, n).extract::<0>(),
            );
        }
    }
}

#[test]
fn log_n_matches_the_const_form() {
    // Bases 0, 1, 2, 10 are special cases, 3..=32 are table entries, and the rest is the fallback.
    let bases: [u32; 14] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20];
    for &n in &bases {
        for &x in &roots_grid() {
            let d = D::splat(x);
            same_bits_f64(
                "log_n/Precision",
                n,
                x,
                d.log_n_p::<Precision>(n).extract::<0>(),
                const_form!(d, log_n_n_p, Precision, n).extract::<0>(),
            );
            same_bits_f64(
                "log_n/Performance",
                n,
                x,
                d.log_n_p::<Performance>(n).extract::<0>(),
                const_form!(d, log_n_n_p, Performance, n).extract::<0>(),
            );

            let xf = x as f32;
            let f = F::splat(xf);
            same_bits_f32(
                "log_n/f32",
                n,
                xf,
                f.log_n_p::<Precision>(n).extract::<0>(),
                const_form!(f, log_n_n_p, Precision, n).extract::<0>(),
            );
        }
    }
}

#[test]
fn log_n_past_the_table_matches_the_const_fallback() {
    let d = D::splat(1234.5);
    same_bits_f64(
        "log_n/33",
        33,
        1234.5,
        d.log_n_p::<Precision>(33).extract::<0>(),
        d.log_n_n_p::<Precision, 33>().extract::<0>(),
    );
    same_bits_f64(
        "log_n/100",
        100,
        1234.5,
        d.log_n_p::<Precision>(100).extract::<0>(),
        d.log_n_n_p::<Precision, 100>().extract::<0>(),
    );
}

/// Points in `[0, 1]` plus the endpoints. The inverse's Newton iteration brackets a root
/// on `[0, 1]` and asserts it, in both forms alike, so the grid stays inside.
fn unit_grid() -> Vec<f64> {
    let mut xs = vec![0.0, 1.0, 0.5];
    let mut t = 0.0;
    while t <= 1.0 {
        xs.push(t);
        t += 1.0 / 37.0;
    }
    xs
}

/// The smoothstep family's runtime forms are a ladder over `0..=4`, by design: the
/// polynomial is a handful of FMAs and nothing table-driven competes with the folded form.
const SMOOTHSTEP_DEGREES: [u32; 5] = [0, 1, 2, 3, 4];

#[test]
fn smoothstep_family_matches_the_const_form() {
    let edges = [None, Some((-1.0, 3.0))];

    for &n in &SMOOTHSTEP_DEGREES {
        for &x in &unit_grid() {
            for e in edges {
                let d = D::splat(x);
                let ed = e.map(|(a, b)| (D::splat(a), D::splat(b)));
                let xd = e.map_or(x, |(a, b)| a + (b - a) * x);
                let dd = D::splat(xd);

                same_bits_f64(
                    "smoothstep",
                    n,
                    xd,
                    dd.smoothstep_p::<Precision>(ed, n).extract::<0>(),
                    const_form!(dd, smoothstep_n_p, Precision, n, ed).extract::<0>(),
                );
                same_bits_f64(
                    "smoothstep_derivative",
                    n,
                    xd,
                    dd.smoothstep_derivative_p::<Precision>(ed, n).extract::<0>(),
                    const_form!(dd, smoothstep_derivative_n_p, Precision, n, ed).extract::<0>(),
                );
                // The inverse takes the value as input, so it is fed the unit grid directly.
                // Not at exactly 0: for n >= 3 both forms hand `[0, 1]` to `newtons_method`
                // as a bracket, whose debug precondition wants f(min) strictly negative, and
                // f(0) = -y is not. A pre-existing sharp edge of the const form, shared
                // exactly, and not this test's to paper over.
                if x != 0.0 {
                    same_bits_f64(
                        "inverse_smoothstep",
                        n,
                        x,
                        d.inverse_smoothstep_p::<Precision>(ed, n).extract::<0>(),
                        const_form!(d, inverse_smoothstep_n_p, Precision, n, ed).extract::<0>(),
                    );
                }

                let f = F::splat(x as f32);
                let ef = e.map(|(a, b)| (F::splat(a as f32), F::splat(b as f32)));
                let ff = F::splat(xd as f32);
                same_bits_f32(
                    "smoothstep/f32",
                    n,
                    xd as f32,
                    ff.smoothstep_p::<Performance>(ef, n).extract::<0>(),
                    const_form!(ff, smoothstep_n_p, Performance, n, ef).extract::<0>(),
                );
                if x != 0.0 {
                    same_bits_f32(
                        "inverse_smoothstep/f32",
                        n,
                        x as f32,
                        f.inverse_smoothstep_p::<Performance>(ef, n).extract::<0>(),
                        const_form!(f, inverse_smoothstep_n_p, Performance, n, ef).extract::<0>(),
                    );
                }
            }
        }
    }
}

#[test]
fn smoothstep_past_the_ladder_is_nan() {
    let d = D::splat(0.5);
    assert!(d.smoothstep_p::<Precision>(None, 5).extract::<0>().is_nan());
    assert!(d.smoothstep_derivative_p::<Precision>(None, 5).extract::<0>().is_nan());
    assert!(d.inverse_smoothstep_p::<Precision>(None, 5).extract::<0>().is_nan());
    // And the top rung is live.
    assert!(!d.smoothstep_p::<Precision>(None, 4).extract::<0>().is_nan());
}
