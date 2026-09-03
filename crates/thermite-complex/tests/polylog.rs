//! The complex polylogarithm against mpmath, through the public `SpecialMath::polylog`
//! entry on `Complex<Vector<f64>>`. References are thermite-special's
//! `tests/polylog_ref/table.rs` (mpmath, 40 digits): every row, real and complex.
//!
//! Accuracy is graded **normwise**: `|got - want| / |want|` with complex moduli. On the cut
//! the real part can be a millionth of the imaginary part (`s = 1 + 1e-6`, `z = 2`), and
//! nothing short of arbitrary precision holds it componentwise.
#![cfg(feature = "special")]
// The generated table carries values that happen to be ln 2 and the like.
#![allow(clippy::excessive_precision, clippy::approx_constant)]

use thermite::math::policy::policies::Precision;
use thermite::prelude::*;
use thermite_complex::Complex;
use thermite_special::{PolylogOrder, SpecialMathWithPolicy};

type V = Vector<f64>;
type C = Complex<V>;

include!("../../thermite-special/tests/polylog_ref/table.rs");

fn order(s: f64) -> PolylogOrder<f64, i64> {
    if s == s.round() {
        PolylogOrder::Integer(s as i64)
    } else {
        PolylogOrder::Real(s)
    }
}

/// The complex vector's order carries its complex element, so a real order is `s + 0i`.
fn corder(s: f64) -> PolylogOrder<Complex<f64>, i64> {
    match order(s) {
        PolylogOrder::Integer(n) => PolylogOrder::Integer(n),
        PolylogOrder::Real(s) => PolylogOrder::Real(Complex::new(s, 0.0)),
    }
}

fn li(s: f64, re: f64, im: f64) -> (f64, f64) {
    let r = C::new(V::splat(re), V::splat(im)).polylog_p::<Precision>(corder(s));
    (r.re.extract::<0>(), r.im.extract::<0>())
}

fn norm_rel(got: (f64, f64), want: (f64, f64)) -> f64 {
    let d = ((got.0 - want.0).powi(2) + (got.1 - want.1).powi(2)).sqrt();
    let n = (want.0 * want.0 + want.1 * want.1).sqrt();
    if n == 0.0 { d } else { d / n }
}

#[test]
fn complex_polylog_matches_mpmath() {
    let mut worst_int = (0.0f64, 0.0, 0.0, 0.0);
    let mut worst_real = (0.0f64, 0.0, 0.0, 0.0);
    let mut bad = 0usize;
    for &(s, zr, zi, wr, wi) in POLYLOG.iter() {
        // A bare real is the value from below: pass it as `z - 0i`.
        let zi_in = if zi == 0.0 && zr > 1.0 { -0.0 } else { zi };
        let got = li(s, zr, zi_in);
        let want = (wr, wi);
        // Tiny references (exact zeros of Li_{-2k}(-1), and the like) are judged absolutely.
        if (wr * wr + wi * wi).sqrt() < 1e-12 {
            let d = ((got.0 - wr).powi(2) + (got.1 - wi).powi(2)).sqrt();
            assert!(d < 1e-13, "near-zero s={s} z={zr}+{zi}i: got {got:?} want {want:?}");
            continue;
        }
        let e = norm_rel(got, want);
        let integer = s == s.round();
        let gate = if integer { 1e-12 } else { 1e-11 };
        if e.is_nan() || e > gate {
            bad += 1;
            if bad <= 40 {
                std::println!(
                    "  OVER s={s:<12} z={zr:e}+{zi:e}i got ({:.16e}, {:.16e}) want ({wr:.16e}, {wi:.16e}) rel {e:.2e}",
                    got.0,
                    got.1
                );
            }
        }
        let w = if integer { &mut worst_int } else { &mut worst_real };
        if e > w.0 {
            *w = (e, s, zr, zi);
        }
    }
    std::println!(
        "  worst integer order: {:.2e} at s={} z={:e}+{:e}i",
        worst_int.0,
        worst_int.1,
        worst_int.2,
        worst_int.3
    );
    std::println!(
        "  worst real order:    {:.2e} at s={} z={:e}+{:e}i",
        worst_real.0,
        worst_real.1,
        worst_real.2,
        worst_real.3
    );
    assert_eq!(bad, 0, "{bad} rows over the gate");
}

/// The two sides of the cut: `+0` is the limit from above and the conjugate of the value
/// from below, for every real order.
#[test]
fn complex_polylog_cut_sides() {
    for &s in &[2.0, 3.0, 0.5, 2.5, -1.5, 7.0] {
        for &x in &[1.5, 3.0, 30.0, 1e4] {
            let below = li(s, x, -0.0);
            let above = li(s, x, 0.0);
            assert!(
                (below.0 - above.0).abs() <= 1e-12 * below.0.abs().max(1.0),
                "re s={s} x={x}: {below:?} {above:?}"
            );
            assert!(
                (below.1 + above.1).abs() <= 1e-12 * below.1.abs().max(1.0),
                "im s={s} x={x}: {below:?} {above:?}"
            );
            // Wood 3.1: Im from below is -pi (ln x)^{s-1} / Gamma(s).
            use thermite_special::ScalarSpecialMath;
            let expect = -std::f64::consts::PI * x.ln().powf(s - 1.0) / s.scalar_tgamma();
            assert!(
                (below.1 - expect).abs() <= 1e-11 * expect.abs().max(1e-300),
                "wood s={s} x={x}: {} vs {expect}",
                below.1
            );
        }
    }
}

/// Real-vector and complex-vector kernels agree on the real part, on and off the cut.
#[test]
fn complex_polylog_agrees_with_real_kernel() {
    for &s in &[2.0, 4.0, -3.0, 0.5, 2.001, -1.5, 7.5] {
        for &x in &[-1e5, -30.0, -2.0, -0.7, -0.1, 0.1, 0.7, 0.999, 1.001, 3.0, 30.0, 1e5] {
            let real = V::splat(x).polylog_p::<Precision>(order(s)).extract::<0>();
            let (cre, cim) = li(s, x, -0.0);
            let scale = real.abs().max(cim.abs()).max(1e-300);
            assert!(
                (real - cre).abs() <= 1e-11 * scale,
                "s={s} x={x}: real {real:e} complex {cre:e}"
            );
        }
    }
}
