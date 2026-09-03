//! Polygamma `psi_n` for `n >= 2` against mpmath, on the positive axis (the reflection
//! region is not implemented yet and is pinned to NaN here). References from
//! `scripts/polygamma_ref.py`. The recurrence identity below is asserted against mpmath
//! by that script before it writes the table.
#![cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "wasm32", feature = "wasm"),
    target_arch = "aarch64"
))]

use thermite::math::policy::policies::Precision;
use thermite::prelude::*;
use thermite_special::SpecialMathWithPolicy;

type D = Vector<f64>;
type F = Vector<f32>;

include!("polygamma_ref/table.rs");

fn pg64(x: f64, n: u32) -> f64 {
    D::splat(x).polygamma_p::<Precision>(n).extract::<0>()
}

fn pg32(x: f32, n: u32) -> f32 {
    F::splat(x).polygamma_p::<Precision>(n).extract::<0>()
}

/// True where the kernel's direct `x^(n+1)` (evaluated at up to `max(x, transition
/// point)`, `transition = d4d + 4n`) exceeds the format's decimal exponent budget.
fn overflows_direct(n: u32, x: f64, d4d: u32, decades: f64) -> bool {
    let transition = (d4d + 4 * n) as f64;
    // Negative arguments reflect to 1 - x before any power is formed.
    let xw = if x > 0.0 { x } else { 1.0 - x };
    (n as f64 + 1.0) * xw.max(transition).log10() > decades
}

fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        return if got == 0.0 { 0.0 } else { f64::INFINITY };
    }
    ((got - want) / want).abs()
}

#[test]
fn polygamma_f64_matches_mpmath() {
    let mut worst = 0.0f64;
    let mut bad = 0usize;
    let mut skipped = 0usize;
    for &(n, x, want) in POLYGAMMA.iter() {
        // Documented domain: past it, x^n (formed directly, at up to max(x, transition
        // point)) overflows while the true value is still representable. The log-domain
        // rescue is queued work, see the kernel docs.
        if overflows_direct(n, x, 6, 300.0) {
            skipped += 1;
            continue;
        }
        let got = pg64(x, n);
        let e = rel(got, want);
        if e > 5e-14 {
            bad += 1;
            std::println!("f64 psi_{n}({x}): got {got:e} want {want:e}, rel err {e:e}");
        }
        worst = worst.max(e);
    }
    std::println!(
        "f64 sweep: worst rel err {worst:e} over {} rows ({skipped} outside the direct-power domain)",
        POLYGAMMA.len() - skipped
    );
    assert!(bad == 0, "{bad} rows over the 5e-14 gate, worst {worst:e}");
}

#[test]
fn polygamma_f32_matches_mpmath() {
    let mut worst = 0.0f64;
    let mut rows = 0usize;
    for &(n, x, want) in POLYGAMMA.iter() {
        // Only arguments exact in f32 (so no condition-number term), orders whose n!
        // is finite in f32, values inside f32's normal range, and the kernel's
        // documented no-overflow domain (see the f64 sweep).
        if x as f32 as f64 != x
            || n >= 35
            || want.abs() > f32::MAX as f64
            || want.abs() < 1e-36
            || overflows_direct(n, x, 2, 36.0)
        {
            continue;
        }
        let got = pg32(x as f32, n) as f64;
        let e = rel(got, want);
        assert!(e <= 2e-6, "f32 psi_{n}({x}): got {got:e} want {want:e}, rel err {e:e}");
        rows += 1;
        worst = worst.max(e);
    }
    std::println!("f32 sweep: worst rel err {worst:e} over {rows} rows");
}

#[test]
fn polygamma_delegates_low_orders() {
    use thermite_special::SpecialMathWithPolicy;

    // The publicly-declared digamma/trigamma (trigamma is public as of the polygamma
    // arc) and polygamma(0)/(1) must be the same code path, bit for bit.
    for x in [0.35f64, 1.0, 2.5, 17.0, -2.25] {
        let v = D::splat(x);
        let d0 = pg64(x, 0);
        let d0_want = v.digamma_p::<Precision>().extract::<0>();
        let d1 = pg64(x, 1);
        let d1_want = v.trigamma_p::<Precision>().extract::<0>();
        assert!(
            d0.to_bits() == d0_want.to_bits() && d1.to_bits() == d1_want.to_bits(),
            "delegation at x={x}: psi_0 {d0:e} vs {d0_want:e}, psi_1 {d1:e} vs {d1_want:e}"
        );
    }
}

#[test]
fn polygamma_recurrence_identity() {
    // psi_n(x + 1) = psi_n(x) + (-1)^n n! x^-(n+1), checked with both sides evaluated by
    // the kernel so it exercises internal consistency across the walk boundary.
    for &n in &[2u32, 3, 5, 10] {
        let mut fac = 1.0f64;
        for k in 2..=n {
            fac *= k as f64;
        }
        for &x in &[0.5f64, 1.0, 2.0, 4.5, 9.0, 30.0] {
            let lhs = pg64(x + 1.0, n);
            let step = if n & 1 == 1 { -fac } else { fac } * x.powi(-(n as i32 + 1));
            let big = pg64(x, n);
            let rhs = big + step;
            // The identity cancels catastrophically at small x (psi_n(x) and the step are
            // each ~n!/x^(n+1) while the difference is psi_n(x+1)), so the tolerance must
            // scale with the size of what cancels.
            let tol = 1e-13 * (1.0 + (big.abs() + step.abs()) / lhs.abs());
            let e = rel(lhs, rhs);
            assert!(
                e <= tol,
                "recurrence psi_{n} at x={x}: {lhs:e} vs {rhs:e}, rel {e:e} (tol {tol:e})"
            );
        }
    }
}

#[test]
fn polygamma_edge_inputs() {
    for &n in &[2u32, 7] {
        assert!(pg64(f64::NAN, n).is_nan(), "psi_{n}(NaN)");
        assert_eq!(pg64(f64::INFINITY, n).abs(), 0.0, "psi_{n}(+inf) -> 0");
    }

    // Poles (zero and the negative integers): odd n has the definite two-sided limit
    // +inf. Even n diverges with opposite signs and Precision checks overflow -> NaN.
    for x in [0.0f64, -1.0, -6.0] {
        assert_eq!(pg64(x, 3), f64::INFINITY, "psi_3({x}) pole");
        assert_eq!(pg64(x, 7), f64::INFINITY, "psi_7({x}) pole");
        assert!(pg64(x, 2).is_nan(), "psi_2({x}) pole should be NaN under Precision");
    }

    // Reflection past the cot-pi table's reach (n > 20) is unimplemented: NaN on the
    // negative axis only.
    assert!(pg64(-2.5, 25).is_nan(), "psi_25(-2.5) past the reflection table");
    assert!(pg64(2.5, 25).is_finite(), "psi_25(2.5) positive axis unaffected");
}

#[test]
fn polygamma_scalar_surface() {
    use thermite_special::ScalarSpecialMathWithPolicy;

    // The scalar wrapper is the width-1 vector path. One row of agreement pins the
    // wrap/unwrap plumbing.
    for &(n, x, _) in POLYGAMMA.iter().take(8) {
        let s = x.scalar_polygamma_p::<Precision>(n);
        let v = pg64(x, n);
        assert!(s.to_bits() == v.to_bits(), "scalar psi_{n}({x}): {s:e} vs vector {v:e}");
    }
}
