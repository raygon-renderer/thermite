//! Division endpoints must stay faithfully rounded, since every widening tier assumes the
//! primitive is within half an ulp.
//!
//! Under `algebraic-scalar`, `add`/`sub`/`mul` are still correctly rounded per operation.
//! `div` is not: `arcp` lets LLVM rewrite `x / c` as `x * fl(1/c)` for a constant `c`,
//! two roundings, measured up to 1.204 ulp at `c = 49.0` (0.490 in an ordinary build).
//! One ulp of bump does not cover that, and enclosure failed for 978 of 20000 numerators.
//!
//! `div_interval` now divides through `round::two_quot(a, b).0`, so these tests measure
//! `two_quot`, not `/`. Swapping back to `/` makes
//! `division_endpoints_are_faithfully_rounded` fail at 1.204 ulp under the feature.
//!
//! The divisors must stay literals: `arcp` only fires on a compile-time constant, and a
//! runtime array disarms the whole file. Scalar backend only; SIMD backends divide with a
//! real instruction.

use thermite::prelude::*;
use thermite_compensated::ScalarValue;

use thermite_interval::{Balanced, Interval, Tightest, WideningPolicy};

type V1 = Vector<f64>;
type I<W> = Interval<V1, W>;

/// Exact `a / b` as a double-double `(q, r)`.
///
/// `#[inline(never)]` and `black_box` are both required: otherwise `b` constant-propagates
/// and `arcp` rewrites the reference division too, so the error measures as zero.
#[inline(never)]
fn exact_div(a: f64, b: f64) -> (f64, f64) {
    let a = core::hint::black_box(a);
    let b = core::hint::black_box(b);

    let q = a / b;
    let (p, e) = ScalarValue::two_prod(V1::splat(q), V1::splat(b));
    let (p, e) = (p.extract::<0>(), e.extract::<0>());
    let (r, _) = ScalarValue::two_diff(V1::splat(a), V1::splat(p));
    let r = (r.extract::<0>() - e) / b;

    (q, r)
}

/// Distance from the correctly-rounded quotient, in ulps.
fn err_ulps(got: f64, a: f64, b: f64) -> f64 {
    let (q, r) = exact_div(a, b);
    let ulp = f64::from_bits(q.abs().to_bits() + 1) - q.abs();
    (((got - q) / ulp) - (r / ulp)).abs()
}

fn encloses<W: WideningPolicy>(q: I<W>, a: f64, b: f64) -> bool {
    let (lo, hi) = (q.lo().extract::<0>(), q.hi().extract::<0>());
    let (tq, tr) = exact_div(a, b);

    (lo < tq || (lo == tq && tr >= 0.0)) && (hi > tq || (hi == tq && tr <= 0.0))
}

/// Runtime numerators. The divisor must stay a literal, see the module docs.
fn numerators() -> impl Iterator<Item = f64> {
    let mut x = 1.0f64;
    (0..20000u32).map(move |i| {
        x = x * 1.0000001 + (i % 7) as f64 * 1e-3;
        if i % 2 == 0 { x } else { -x }
    })
}

macro_rules! check_divisors {
    ($($d:literal),* $(,)?) => {{
        let mut worst = 0.0f64;
        let mut bad: u32 = 0;
        $({
            for a in numerators() {
                // The primitive `div_interval` uses. A bare `/` measures 1.204 ulp here.
                let got = ScalarValue::two_quot(V1::splat(a), V1::splat($d)).0;
                let e = err_ulps(got.extract::<0>(), a, $d);
                if e > worst {
                    worst = e;
                }

                let x: I<Balanced> = Interval::bounds(V1::splat(a), V1::splat(a));
                let y: I<Balanced> = Interval::bounds(V1::splat($d), V1::splat($d));
                if !encloses(x / y, a, $d) {
                    bad += 1;
                }

                let xt: I<Tightest> = Interval::bounds(V1::splat(a), V1::splat(a));
                let yt: I<Tightest> = Interval::bounds(V1::splat($d), V1::splat($d));
                if !encloses(xt / yt, a, $d) {
                    bad += 1;
                }
            }
        })*
        (worst, bad)
    }};
}

/// A division endpoint is within half an ulp. The direct check; the containment suite
/// only tests the conclusion, which degrades silently and then fails all at once.
#[test]
fn division_endpoints_are_faithfully_rounded() {
    let (worst, _) = check_divisors!(2.0, 3.0, 7.0, 10.0, 49.0, 0.1, 1.0e3, 6.283185307179586);

    assert!(
        worst <= 0.5,
        "a division endpoint is {worst:.3} ulp from the correctly-rounded quotient; every \
         widening tier assumes <= 0.5, so enclosure is no longer guaranteed. Under \
         `algebraic-scalar` this is `arcp` rewriting `x / c` as `x * fl(1/c)`."
    );
}

/// The consequence, separate so a failure says which layer broke.
#[test]
fn division_encloses_the_true_quotient() {
    let (_, bad) = check_divisors!(2.0, 3.0, 7.0, 10.0, 49.0, 0.1, 1.0e3, 6.283185307179586);

    assert_eq!(bad, 0, "{bad} division enclosures do not contain the true quotient");
}
