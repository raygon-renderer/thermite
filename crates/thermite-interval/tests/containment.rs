//! THE law: for every operation `F` lifting
//! a real function `f`, and every input interval `X` with sample points
//! `x in X`, assert `F(X).contains(f(x))`, with `f` referenced exactly (the
//! double-double result of thermite-compensated's error-free transforms).
//!
//! Runs at `Vector<f64>` (the width-1 scalar seed) across all three widening
//! policies. Tightness is checked ordinally (Tightest <= Balanced <= Fastest
//! on accumulated width), never as absolute numbers.

use thermite::prelude::*;
use thermite::vector::ops::MulAddExt;
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
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
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
            assert!(
                contains_exact(x + y, s, r),
                "add: {xs:e} + {ys:e} not in {:?}",
                bounds(x + y)
            );
        }
        let (s, r) = s_two_sum(xs, -ys);
        if s.is_finite() {
            assert!(
                contains_exact(x - y, s, r),
                "sub: {xs:e} - {ys:e} not in {:?}",
                bounds(x - y)
            );
        }

        // mul
        let (p, r) = s_two_prod(xs, ys);
        if p.is_finite() {
            assert!(
                contains_exact(x * y, p, r),
                "mul: {xs:e} * {ys:e} not in {:?}",
                bounds(x * y)
            );
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
            assert!(
                lo <= q && q <= hi,
                "div: {xs:e} / {ys:e} = {q:e} not in [{lo:e}, {hi:e}]"
            );
        }

        // sqrt (correctly rounded reference)
        if xs >= 0.0 {
            let s = xs.sqrt();
            let (lo, hi) = bounds(x.abs_interval().sqrt_interval());
            let sa = xs.abs().sqrt();
            let _ = s;
            assert!(
                lo <= sa && sa <= hi,
                "sqrt: sqrt({:e}) = {sa:e} not in [{lo:e}, {hi:e}]",
                xs.abs()
            );
        }

        // fma: one interval operation, referenced against the exact `xs*ys + zs`.
        // `fl(fl(xs*ys) + zs)` is DOUBLE rounded, and under cancellation it lands
        // several ulps from the exact value, which is not a normalized double-double,
        // and `le_exact`/`ge_exact` need one. So renormalize the head against the two
        // residuals before comparing, or a sound bound reads as out of range.
        let (z, zs) = rand_interval::<W>(&mut state);
        let (p, r1) = s_two_prod(xs, ys);
        let (s, r2) = s_two_sum(p, zs);
        let (v, e) = s_two_sum(s, r2 + r1);
        if v.is_finite() {
            let got = MulAddExt::mul_add(x, y, z);
            assert!(
                contains_exact(got, v, e),
                "fma: {xs:e} * {ys:e} + {zs:e} not in {:?}",
                bounds(got)
            );
            // The whole point of fusing: never wider than multiplying and adding.
            let (flo, fhi) = bounds(got);
            let (clo, chi) = bounds(x * y + z);
            assert!(
                flo >= clo && fhi <= chi,
                "fma wider than mul-then-add: [{flo:e}, {fhi:e}] vs [{clo:e}, {chi:e}]"
            );
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
    // the scalar test backend HAS_NATIVE_FMA may be False, so only assert
    // containment there.
    let (lo, hi) = bounds(xb * iv(2.0, 2.0));
    assert!(lo <= 3.0 && hi >= 3.0);
}

/// Cancellation is the case the fused FMA exists for. This triple (from the sweep)
/// forms a product near `-4.5e21` that cancels down to `-2.3e20`. Multiplying first
/// commits the product's rounding at the LARGER scale (about 7 ulps of the result),
/// and no later step wins that back. One fused rounding never pays it.
#[test]
fn the_fused_fma_survives_cancellation() {
    let x: I<Balanced> = iv(-5.558651027089484e-10, -5.558651027089484e-10);
    let y: I<Balanced> = iv(8.216029136540841e30, 8.216029136540841e30);
    let z: I<Balanced> = iv(4.3318161865435654e21, 4.3318161865435654e21);

    let (flo, fhi) = bounds(MulAddExt::mul_add(x, y, z));
    let (clo, chi) = bounds(x * y + z);

    // The exact value, as a renormalized double-double.
    let (p, r1) = s_two_prod(-5.558651027089484e-10, 8.216029136540841e30);
    let (s, r2) = s_two_sum(p, 4.3318161865435654e21);
    let (v, e) = s_two_sum(s, r2 + r1);

    assert!(le_exact(flo, v, e) && ge_exact(fhi, v, e), "fma must enclose");
    assert!(fhi - flo <= chi - clo, "fma must not be wider than mul-then-add");

    // Where the inner vector actually fuses, the product's rounding never enters.
    if matches!(<I<Balanced> as MulAddExt>::HAS_NATIVE_FMA, thermite::tribool::True) {
        assert!(
            (fhi - flo) * 4.0 < chi - clo,
            "fused [{flo:e}, {fhi:e}] against composed [{clo:e}, {chi:e}]"
        );
    }
}

/// `0 * inf` is the set limit 0 in the fused FMA exactly as in the multiply, so that
/// corner contributes the addend rather than a NaN that eats the min.
#[test]
fn the_fma_reads_zero_times_infinity_as_the_set_limit() {
    let inf = f64::INFINITY;

    // Every product is exactly zero, so the result is exactly the addend.
    let r: I<Balanced> = MulAddExt::mul_add(iv(0.0, 0.0), iv(1.0, inf), iv(2.0, 3.0));
    assert_eq!(bounds(r), (2.0, 3.0), "a zero factor gives back the addend");

    // [0, 2] * [1, inf] spans [0, inf], so the sum spans [5, inf]: the NaN corner
    // has to read as 5, one ulp of widening aside.
    let r: I<Balanced> = MulAddExt::mul_add(iv(0.0, 2.0), iv(1.0, inf), iv(5.0, 5.0));
    let (lo, hi) = bounds(r);
    assert!(lo <= 5.0 && lo > 4.99 && hi == inf, "got [{lo:e}, {hi:e}]");
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

    assert!(
        t <= b && b <= f,
        "width order violated: tightest={t:e} balanced={b:e} fastest={f:e}"
    );
    assert!(t > 0.0, "512 inexact ops cannot leave a degenerate interval");
}

/// Empty lanes are absorbing through arithmetic.
#[test]
fn empty_propagates() {
    let e: I<Balanced> = Interval::empty();
    let x: I<Balanced> = iv(1.0, 2.0);

    for r in [
        e + x,
        x + e,
        e * x,
        x * e,
        e / x,
        x / e,
        e.sqrt_interval(),
        e.square_interval(),
        MulAddExt::mul_add(e, x, x),
        MulAddExt::mul_add(x, e, x),
        MulAddExt::mul_add(x, x, e),
    ] {
        assert!(r.is_empty().all(), "empty must stay empty: {:?}", bounds(r));
    }
}

/// Regression: `[+inf, -inf] + [-inf, hi]` hit `inf + -inf = NaN`, and a NaN
/// bound is not `lo > hi`, so the empty lane silently un-emptied.
#[test]
fn empty_plus_infinite_endpoint_stays_empty() {
    fn check<W: WideningPolicy>() {
        let e: I<W> = Interval::empty();
        for x in [Interval::entire(), iv(f64::NEG_INFINITY, 1.0), iv(1.0, f64::INFINITY)] {
            for r in [e + x, x + e, e - x, x - e] {
                assert!(r.is_empty().all(), "empty must stay empty: {:?}", bounds(r));
            }
        }
    }
    check::<Fastest>();
    check::<Balanced>();
    check::<Tightest>();
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
fn set_ops() {
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

// --- difference_of_products / sum_of_products --------------------------------
//
// These carry a `SpecializedCoreMath` override (see `src/math.rs`). The trait
// default compensates above `Average` precision by recovering the rounding that
// `c * d` discarded and adding it back. For an interval there is no such discarded
// rounding to recover (`c * d` is already an outward-rounded enclosure), so the
// default's correction degenerates to `cd - cd`, a symmetric band of twice the
// product's width, and adding it back inflates the result 2x for nothing.
//
// Two things are pinned: THE law still holds (it is an enclosure), and the result
// does not move with the precision policy, which is what would fail if the override
// were dropped and the default's compensation came back.

/// Enclosure, plus policy-independence, for `ab - cd` and `ab + cd`.
#[test]
fn products_enclose_and_ignore_precision_policy() {
    use thermite::math::policy::policies::{HighPerformance, Performance, Precision};
    use thermite::math::{CoreMath, CoreMathWithPolicy};

    // Degenerate (point) intervals: the exact result is then a single real, so the
    // enclosure check is sharp rather than trivially satisfied by a wide input.
    let cases: &[(f64, f64, f64, f64)] = &[
        (0.1, 0.3, 0.3, 0.1),                       // exactly-equal products
        (33962.035, -30438.8, 41563.4, -24871.969), // Kahan's cancelling case
        (1.0 / 3.0, 7.0 / 11.0, 1e-8, 3.7e12),
        (-2.718281828459045, 3.141592653589793, 0.1, 0.3),
    ];

    fn point<W: WideningPolicy>(x: f64) -> I<W> {
        Interval::bounds(V1::splat(x), V1::splat(x))
    }

    for &(a, b, c, d) in cases {
        // Exact `a*b - c*d` as a double-double, via error-free transforms.
        let (p, pe) = s_two_prod(a, b);
        let (q, qe) = s_two_prod(c, d);
        let (diff, de) = s_two_sum(p, -q);
        let (sum, se) = s_two_sum(p, q);

        let (ai, bi, ci, di) = (
            point::<Tightest>(a),
            point::<Tightest>(b),
            point::<Tightest>(c),
            point::<Tightest>(d),
        );

        let dop = ai.difference_of_products(bi, ci, di);
        let sop = ai.sum_of_products(bi, ci, di);

        assert!(
            contains_exact(dop, diff, de + pe - qe),
            "difference_of_products({a:e}, {b:e}, {c:e}, {d:e}) = {:?} excludes {diff:e}",
            bounds(dop)
        );
        assert!(
            contains_exact(sop, sum, se + pe + qe),
            "sum_of_products({a:e}, {b:e}, {c:e}, {d:e}) = {:?} excludes {sum:e}",
            bounds(sop)
        );

        // Policy-independent: identical bounds either side of the Average boundary
        // the trait default keys on.
        for (name, got) in [
            ("Performance", ai.difference_of_products_p::<Performance>(bi, ci, di)),
            ("Precision", ai.difference_of_products_p::<Precision>(bi, ci, di)),
            (
                "HighPerformance",
                ai.difference_of_products_p::<HighPerformance>(bi, ci, di),
            ),
        ] {
            assert_eq!(
                bounds(got),
                bounds(dop),
                "difference_of_products moved with policy {name} on ({a:e}, {b:e}, {c:e}, {d:e})"
            );
        }
    }
}
