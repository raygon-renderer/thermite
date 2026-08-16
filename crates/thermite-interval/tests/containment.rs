//! THE law: for every operation `F` lifting
//! a real function `f`, and every input interval `X` with sample points
//! `x in X`, assert `F(X).contains(f(x))`, with `f` referenced exactly (the
//! double-double result of thermite-compensated's error-free transforms).
//!
//! Runs at `Vector<f64>` (the width-1 scalar seed) across all three widening
//! policies. Tightness is checked ordinally (Tightest <= Balanced <= Fastest
//! on accumulated width), never as absolute numbers.

use thermite::prelude::*;
use thermite_compensated::ScalarValue;
use thermite_interval::{Balanced, Fastest, Interval, Tightest, WideningPolicy};

type V1 = Vector<f64>;
type I<W> = Interval<V1, W>;

fn iv<W: WideningPolicy>(lo: f64, hi: f64) -> I<W> {
    Interval::bounds(V1::splat(lo), V1::splat(hi))
}

fn bounds<W: WideningPolicy>(i: I<W>) -> (f64, f64) {
    (i.lo().extract::<0>(), i.hi().extract::<0>())
}

/// `bound <= v + e` against an exact double-double value.
fn le_exact(bound: f64, v: f64, e: f64) -> bool {
    bound < v || (bound == v && e >= 0.0)
}
fn ge_exact(bound: f64, v: f64, e: f64) -> bool {
    bound > v || (bound == v && e <= 0.0)
}

fn contains_exact<W: WideningPolicy>(i: I<W>, v: f64, e: f64) -> bool {
    let (lo, hi) = bounds(i);
    le_exact(lo, v, e) && ge_exact(hi, v, e)
}

fn s_two_sum(a: f64, b: f64) -> (f64, f64) {
    let (s, r) = ScalarValue::two_sum(V1::splat(a), V1::splat(b));
    (s.extract::<0>(), r.extract::<0>())
}
fn s_two_prod(a: f64, b: f64) -> (f64, f64) {
    let (p, r) = ScalarValue::two_prod(V1::splat(a), V1::splat(b));
    (p.extract::<0>(), r.extract::<0>())
}

fn lcg(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *state
}

/// Random finite f64 over a wide exponent range, both signs.
fn rand_finite(state: &mut u64) -> f64 {
    let mantissa = lcg(state) >> 12;
    let exp = 500 + (lcg(state) % 1000);
    let sign = lcg(state) & (1 << 63);
    f64::from_bits(sign | (exp << 52) | mantissa)
}

/// A random non-degenerate interval around a random point.
fn rand_interval<W: WideningPolicy>(state: &mut u64) -> (I<W>, f64) {
    let a = rand_finite(state);
    let b = rand_finite(state);
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    // Sample point: one of the endpoints (worst case for containment).
    let x = if lcg(state) & 1 == 0 { lo } else { hi };
    (iv(lo, hi), x)
}

fn containment_sweep<W: WideningPolicy>() {
    let mut state = 0xC0FFEE_u64;

    for _ in 0..100_000 {
        let (x, xs) = rand_interval::<W>(&mut state);
        let (y, ys) = rand_interval::<W>(&mut state);

        // add / sub
        let (s, r) = s_two_sum(xs, ys);
        if s.is_finite() {
            assert!(contains_exact(x + y, s, r), "add: {xs:e} + {ys:e} not in {:?}", bounds(x + y));
        }
        let (s, r) = s_two_sum(xs, -ys);
        if s.is_finite() {
            assert!(contains_exact(x - y, s, r), "sub: {xs:e} - {ys:e} not in {:?}", bounds(x - y));
        }

        // mul
        let (p, r) = s_two_prod(xs, ys);
        if p.is_finite() {
            assert!(contains_exact(x * y, p, r), "mul: {xs:e} * {ys:e} not in {:?}", bounds(x * y));
        }

        // square (the dependency-correct one)
        let (p, r) = s_two_prod(xs, xs);
        if p.is_finite() {
            let sq = thermite::vector::ops::Square::square(x);
            assert!(contains_exact(sq, p, r), "square: {xs:e}^2 not in {:?}", bounds(sq));
        }

        // div: reference via q = fl(xs/ys), residual from fma(q, ys, -xs).
        // q + r/ys is not a clean dd, so check plain containment of the
        // correctly-rounded quotient with a 1-ulp margin on the reference
        // side instead: q is within 0.5 ulp of the true quotient, and the
        // interval is outward-rounded by at least 1 ulp beyond fl.
        let q = xs / ys;
        if q.is_finite() {
            let (lo, hi) = bounds(x / y);
            assert!(lo <= q && q <= hi, "div: {xs:e} / {ys:e} = {q:e} not in [{lo:e}, {hi:e}]");
        }

        // sqrt (correctly rounded reference)
        if xs >= 0.0 {
            let s = xs.sqrt();
            let (lo, hi) = bounds(x.abs_interval().sqrt_interval());
            let sa = xs.abs().sqrt();
            let _ = s;
            assert!(lo <= sa && sa <= hi, "sqrt: sqrt({:e}) = {sa:e} not in [{lo:e}, {hi:e}]", xs.abs());
        }

        // abs, min, max are exact set maps: plain point containment.
        assert!(x.abs_interval().contains(V1::splat(xs.abs())).all());
        assert!(x.min_interval(y).contains(V1::splat(xs.min(ys))).all());
        assert!(x.max_interval(y).contains(V1::splat(xs.max(ys))).all());
    }
}

#[test]
fn containment_fastest() {
    containment_sweep::<Fastest>();
}

#[test]
fn containment_balanced() {
    containment_sweep::<Balanced>();
}

#[test]
fn containment_tightest() {
    containment_sweep::<Tightest>();
}

/// Tightest keeps degenerate intervals degenerate through exact operations.
#[test]
fn tightest_preserves_exactness() {
    let x: I<Tightest> = iv(1.5, 1.5);
    let y: I<Tightest> = iv(2.25, 2.25);

    assert_eq!(bounds(x + y), (3.75, 3.75), "exact add must stay degenerate");
    assert_eq!(bounds(x * iv(2.0, 2.0)), (3.0, 3.0), "mul by power of two is exact");

    // Balanced fattens the add (bump tier)...
    let xb: I<Balanced> = iv(1.5, 1.5);
    let yb: I<Balanced> = iv(2.25, 2.25);
    let (lo, hi) = bounds(xb + yb);
    assert!(lo < 3.75 && hi > 3.75);

    // ...but its residual mul stays exact when hardware FMA is present. On
    // the scalar test backend HAS_TRUE_FMA may be false, so only assert
    // containment there.
    let (lo, hi) = bounds(xb * iv(2.0, 2.0));
    assert!(lo <= 3.0 && hi >= 3.0);
}

/// Accumulated widths order by tier: Tightest <= Balanced <= Fastest.
#[test]
fn width_orders_by_tier() {
    fn chain<W: WideningPolicy>() -> f64 {
        let mut state = 7u64;
        let mut acc: I<W> = iv(1.0, 1.0);
        for _ in 0..512 {
            let c = 0.5 + (lcg(&mut state) >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            acc = acc * iv(0.75, 0.75) + iv(c, c);
        }
        let (lo, hi) = bounds(acc);
        hi - lo
    }

    let f = chain::<Fastest>();
    let b = chain::<Balanced>();
    let t = chain::<Tightest>();

    assert!(t <= b && b <= f, "width order violated: tightest={t:e} balanced={b:e} fastest={f:e}");
    assert!(t > 0.0, "512 inexact ops cannot leave a degenerate interval");
}

/// Empty lanes are absorbing through arithmetic.
#[test]
fn empty_propagates() {
    let e: I<Balanced> = Interval::empty();
    let x: I<Balanced> = iv(1.0, 2.0);

    for r in [e + x, x + e, e * x, x * e, e / x, x / e, e.sqrt_interval(), e.square_interval()] {
        assert!(r.is_empty().all(), "empty must stay empty: {:?}", bounds(r));
    }
}

/// Division by a zero-containing interval is the entire line.
#[test]
fn zero_divisor_is_entire() {
    let x: I<Balanced> = iv(1.0, 2.0);
    let d: I<Balanced> = iv(-0.5, 0.5);
    let (lo, hi) = bounds(x / d);
    assert_eq!(lo, f64::NEG_INFINITY);
    assert_eq!(hi, f64::INFINITY);
}

/// sqrt domain handling: straddling clamps at zero, all-negative is empty.
#[test]
fn sqrt_domain() {
    let s: I<Balanced> = iv(-1.0, 4.0);
    let (lo, hi) = bounds(s.sqrt_interval());
    assert_eq!(lo, 0.0);
    assert!(hi >= 2.0);

    let n: I<Balanced> = iv(-4.0, -1.0);
    assert!(n.sqrt_interval().is_empty().all());
}

/// Zero-straddling square: lower bound is exactly zero, never negative.
#[test]
fn square_straddling_zero() {
    let x: I<Balanced> = iv(-1.0, 2.0);
    let (lo, hi) = bounds(x.square_interval());
    assert_eq!(lo, 0.0, "square is never negative");
    assert!(hi >= 4.0);

    // The dependency problem this avoids: x * x has a negative lower bound.
    let (naive_lo, _) = bounds(x * x);
    assert!(naive_lo < 0.0, "x * x is the wide form (expected here)");
}

/// Overflow: a lower bound that rounds to +inf must collapse to MAX in every
/// tier (bump gets it from IEEE next_down, scale and residual from the clamp).
#[test]
fn overflow_clamps() {
    fn check<W: WideningPolicy>() {
        let m: I<W> = iv(f64::MAX, f64::MAX);
        let (lo, hi) = bounds(m + m);
        assert_eq!(lo, f64::MAX, "lower bound must clamp to MAX");
        assert_eq!(hi, f64::INFINITY);
    }
    check::<Fastest>();
    check::<Balanced>();
    check::<Tightest>();
}

/// Set operations.
#[test]
fn set_ops()
{
    let a: I<Balanced> = iv(0.0, 2.0);
    let b: I<Balanced> = iv(1.0, 3.0);

    assert_eq!(bounds(a.intersect(b)), (1.0, 2.0));
    assert_eq!(bounds(a.hull(b)), (0.0, 3.0));
    assert!(a.intersect(iv(5.0, 6.0)).is_empty().all());
    assert!(a.contains(V1::splat(1.5)).all());
    assert!(!a.contains(V1::splat(2.5)).any());
    assert!(iv::<Balanced>(0.5, 1.5).subset_of(a).all());
    assert!(!b.subset_of(a).any());
}

/// A generic kernel written against FloatVector bounds runs on intervals
/// unchanged and its enclosure contains the true value.
#[test]
fn generic_kernel_containment() {
    // hypot-ish: sqrt(a*a + b*b), spelled through mul_adde like real kernels.
    fn kernel<V: FloatVector>(a: V, b: V) -> V {
        a.mul_adde(a, b * b).sqrt()
    }

    let mut state = 99u64;
    for _ in 0..10_000 {
        let a = 1.0 + (lcg(&mut state) >> 40) as f64 * 1e-4;
        let b = 2.0 + (lcg(&mut state) >> 40) as f64 * 1e-4;

        let enclosed: I<Tightest> = kernel(Interval::degenerate(V1::splat(a)), Interval::degenerate(V1::splat(b)));

        // Reference in double-double: the SAME generic kernel run on
        // Compensated. Two composites, one function, checking each other.
        let ca = thermite_compensated::Compensated::<V1>::new(V1::splat(a));
        let cb = thermite_compensated::Compensated::<V1>::new(V1::splat(b));
        let reference = kernel(ca, cb);
        let (rv, re) = (reference.value().extract::<0>(), reference.error().extract::<0>());

        assert!(
            contains_exact(enclosed, rv, re),
            "kernel({a}, {b}): dd {rv:e}+{re:e} not in {:?}",
            bounds(enclosed)
        );
    }
}

/// PrimalProjection: degenerate embedding, midpoint projection.
#[test]
fn primal_projection() {
    use thermite::math::PrimalProjection;

    let p = V1::splat(3.5);
    let i: I<Balanced> = Interval::from_primal(p);
    assert_eq!(bounds(i), (3.5, 3.5));
    assert_eq!(i.to_primal().extract::<0>(), 3.5);
}
