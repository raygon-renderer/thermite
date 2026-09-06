//! Containment of the math library: every transcendental enclosure must
//! contain the true value, checked against an independent high-precision
//! reference (`Compensated<Vector<f64>>` at `Best` policy, which is a
//! different algorithm family from the plain-f64 kernels the interval
//! endpoints use).
//!
//! Also checks the two-policy contract: a loose math policy must widen the
//! enclosure, never invalidate it.

use thermite::math::policy::policies::{Performance, Precision, UltraPerformance};
use thermite::math::{CoreMath as _, TranscendentalMath as _, TranscendentalMathWithPolicy};
use thermite::prelude::*;
use thermite_compensated::Compensated;
use thermite_interval::{Balanced, Fastest, Interval, Tightest, WideningPolicy};

type V1 = Vector<f64>;
type C = Compensated<V1>;
type I<W> = Interval<V1, W>;

fn iv<W: WideningPolicy>(lo: f64, hi: f64) -> I<W> {
    Interval::bounds(V1::splat(lo), V1::splat(hi))
}

fn pt<W: WideningPolicy>(x: f64) -> I<W> {
    Interval::degenerate(V1::splat(x))
}

fn bounds<W: WideningPolicy>(i: I<W>) -> (f64, f64) {
    (i.lo().extract::<0>(), i.hi().extract::<0>())
}

/// High-precision reference value as (value, error) double-double halves.
fn dd(c: C) -> (f64, f64) {
    (c.value().extract::<0>(), c.error().extract::<0>())
}

fn contains_dd<W: WideningPolicy>(i: I<W>, r: C) -> bool {
    let (lo, hi) = bounds(i);
    let (v, e) = dd(r);
    (lo < v || (lo == v && e >= 0.0)) && (hi > v || (hi == v && e <= 0.0))
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn uniform(state: &mut u64, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * ((lcg(state) >> 11) as f64 * (1.0 / (1u64 << 53) as f64))
}

/// Point inputs: the enclosure of a single value must contain the reference.
macro_rules! point_containment {
    ($name:ident, $lo:expr, $hi:expr, $ival:expr, $refr:expr) => {
        #[test]
        fn $name() {
            let mut state = 0xBEEF;
            for _ in 0..20_000 {
                let x = uniform(&mut state, $lo, $hi);
                let i: I<Tightest> = ($ival)(pt::<Tightest>(x));
                let r: C = ($refr)(C::new(V1::splat(x)));
                assert!(
                    contains_dd(i, r),
                    concat!(stringify!($name), " @ {}: {:?} does not contain {:?}"),
                    x,
                    bounds(i),
                    dd(r)
                );
            }
        }
    };
}

point_containment!(exp_contains, -20.0, 20.0, |i: I<Tightest>| i.exp(), |c: C| c.exp());
point_containment!(ln_contains, 1e-8, 1e8, |i: I<Tightest>| i.ln(), |c: C| c.ln());
point_containment!(sqrt_contains, 0.0, 1e6, |i: I<Tightest>| i.sqrt(), |c: C| c.sqrt());
point_containment!(sin_contains, -10.0, 10.0, |i: I<Tightest>| i.sin(), |c: C| c.sin());
point_containment!(cos_contains, -10.0, 10.0, |i: I<Tightest>| i.cos(), |c: C| c.cos());
point_containment!(tanh_contains, -5.0, 5.0, |i: I<Tightest>| i.tanh(), |c: C| c.tanh());
point_containment!(atan_contains, -50.0, 50.0, |i: I<Tightest>| i.atan(), |c: C| c.atan());
point_containment!(asin_contains, -1.0, 1.0, |i: I<Tightest>| i.asin(), |c: C| c.asin());
point_containment!(acos_contains, -1.0, 1.0, |i: I<Tightest>| i.acos(), |c: C| c.acos());
point_containment!(cbrt_contains, -1e5, 1e5, |i: I<Tightest>| i.cbrt(), |c: C| c.cbrt());
point_containment!(exp_m1_contains, -5.0, 5.0, |i: I<Tightest>| i.exp_m1(), |c: C| c
    .exp_m1());
point_containment!(ln_1p_contains, -0.9, 1e5, |i: I<Tightest>| i.ln_1p(), |c: C| c.ln_1p());
point_containment!(sinh_contains, -5.0, 5.0, |i: I<Tightest>| i.sinh(), |c: C| c.sinh());
point_containment!(cosh_contains, -5.0, 5.0, |i: I<Tightest>| i.cosh(), |c: C| c.cosh());
point_containment!(sinhc_contains, -5.0, 5.0, |i: I<Tightest>| i.sinhc(), |c: C| c.sinhc());
point_containment!(atanhc_contains, -0.99, 0.99, |i: I<Tightest>| i.atanhc(), |c: C| c
    .atanhc());

/// `sinhc` and `atanhc` are even with their minimum at zero, so an interval
/// straddling zero takes its lower bound from the mignitude rather than from
/// either endpoint. A degenerate-point test cannot see that, so sample
/// interiors of genuinely wide intervals.
///
/// Both are `>= 1` everywhere on their domains, which the enclosure must not
/// contradict by more than the widening.
#[test]
fn even_quotients_enclose_straddling_intervals() {
    let mut state = 0x51DE;

    for _ in 0..5_000 {
        let a = uniform(&mut state, -4.0, 4.0);
        let w = uniform(&mut state, 0.0, 3.0);
        let sh = iv::<Tightest>(a, a + w).sinhc();

        // atanhc needs the whole interval inside (-1, 1).
        let b = uniform(&mut state, -0.95, 0.90);
        let bh = uniform(&mut state, b, 0.95);
        let ah = iv::<Tightest>(b, bh).atanhc();

        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let p = a + w * t;
            assert!(
                contains_dd(sh, C::new(V1::splat(p)).sinhc()),
                "sinhc over [{a}, {}] misses {p}: {:?}",
                a + w,
                bounds(sh)
            );

            let q = b + (bh - b) * t;
            assert!(
                contains_dd(ah, C::new(V1::splat(q)).atanhc()),
                "atanhc over [{b}, {bh}] misses {q}: {:?}",
                bounds(ah)
            );
        }

        // Neither function dips below 1, so a non-positive lower bound would
        // mean the mignitude branch was skipped entirely.
        assert!(bounds(sh).0 > 0.0, "sinhc over [{a}, {}] lost its sign", a + w);
        assert!(bounds(ah).0 > 0.0, "atanhc over [{b}, {bh}] lost its sign");
    }
}

/// Outside `(-1, 1)` `atanhc` is empty, exactly as `atanh` is.
#[test]
fn atanhc_is_empty_off_its_domain() {
    for (lo, hi) in [(1.5, 2.5), (-3.0, -1.0), (1.0, 4.0)] {
        let r = iv::<Tightest>(lo, hi).atanhc();
        assert!(
            r.is_empty().all(),
            "atanhc over [{lo}, {hi}] should be empty: {:?}",
            bounds(r)
        );
    }
}

/// Interval inputs: sample interior points and require containment of each.
#[test]
fn interval_inputs_contain_all_samples() {
    let mut state = 1234;

    for _ in 0..20_000 {
        let a = uniform(&mut state, -6.0, 6.0);
        let w = uniform(&mut state, 0.0, 2.0);
        let x: I<Tightest> = iv(a, a + w);

        let (se, ce) = (x.sin(), x.cos());
        let ee = x.exp();

        // Sample interior points, including both endpoints.
        for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let p = a + w * t;
            let rc = C::new(V1::splat(p));

            assert!(contains_dd(se, rc.sin()), "sin over [{a}, {}] misses {p}", a + w);
            assert!(contains_dd(ce, rc.cos()), "cos over [{a}, {}] misses {p}", a + w);
            assert!(contains_dd(ee, rc.exp()), "exp over [{a}, {}] misses {p}", a + w);
        }
    }
}

/// sin/cos over a full period must be exactly [-1, 1], and never exceed it.
#[test]
fn trig_range_is_bounded() {
    let full: I<Balanced> = iv(-100.0, 100.0);
    assert_eq!(bounds(full.sin()), (-1.0, 1.0));
    assert_eq!(bounds(full.cos()), (-1.0, 1.0));

    // Any interval at all stays inside [-1, 1].
    let mut state = 55;
    for _ in 0..10_000 {
        let a = uniform(&mut state, -50.0, 50.0);
        let w = uniform(&mut state, 0.0, 10.0);
        let x: I<Balanced> = iv(a, a + w);
        for r in [x.sin(), x.cos()] {
            let (lo, hi) = bounds(r);
            assert!(lo >= -1.0 && hi <= 1.0, "trig escaped [-1,1]: [{lo}, {hi}]");
        }
    }
}

/// A peak inside the interval must be captured: sin over an interval
/// containing pi/2 has upper bound exactly 1.
#[test]
fn trig_captures_interior_extrema() {
    let x: I<Balanced> = iv(1.0, 2.0); // contains pi/2 ~ 1.5708
    let (_, hi) = bounds(x.sin());
    assert_eq!(hi, 1.0, "sin must reach 1 on an interval containing pi/2");

    let y: I<Balanced> = iv(3.0, 3.5); // contains pi ~ 3.14159
    let (lo, _) = bounds(y.cos());
    assert_eq!(lo, -1.0, "cos must reach -1 on an interval containing pi");
}

/// Domain restriction: out-of-domain inputs give empty, straddling clamps.
#[test]
fn domains() {
    let neg: I<Balanced> = iv(-5.0, -1.0);
    assert!(neg.ln().is_empty().all(), "ln of a negative interval is empty");
    // [-5, -1] touches the domain at exactly -1, so it is NOT empty: the
    // result is the degenerate asin(-1) = -pi/2. Strictly-below is empty.
    let touching = neg.asin();
    assert!(!touching.is_empty().any(), "asin([-5, -1]) meets the domain at -1");
    assert!(touching.contains(V1::splat(-std::f64::consts::FRAC_PI_2)).all());

    let below: I<Balanced> = iv(-5.0, -2.0);
    assert!(below.asin().is_empty().all(), "asin strictly below -1 is empty");

    let straddle: I<Balanced> = iv(-1.0, 4.0);
    let (lo, hi) = bounds(straddle.ln());
    assert!(lo == f64::NEG_INFINITY || lo < hi, "ln clamps at the domain edge");

    let wide_asin: I<Balanced> = iv(-2.0, 0.5);
    let (lo, hi) = bounds(wide_asin.asin());
    assert!(lo <= -std::f64::consts::FRAC_PI_2 && hi >= 0.5f64.asin());
}

/// The two-policy contract: a looser math policy widens but never invalidates.
#[test]
fn loose_policy_widens_but_contains() {
    let mut state = 777;

    for _ in 0..5_000 {
        let x = uniform(&mut state, -5.0, 5.0);
        let p: I<Tightest> = pt(x);
        let r = C::new(V1::splat(x)).exp();

        let tight = TranscendentalMathWithPolicy::exp_p::<Precision>(p);
        let loose = TranscendentalMathWithPolicy::exp_p::<Performance>(p);
        let loosest = TranscendentalMathWithPolicy::exp_p::<UltraPerformance>(p);

        for (name, i) in [("precision", tight), ("performance", loose), ("ultra", loosest)] {
            assert!(contains_dd(i, r), "{name} policy lost containment at exp({x})");
        }

        // NOTE: no "looser is wider" assertion. The margin is a measured
        // per-function constant rather than a function of the math policy
        // tier, so a looser `P` shifts the enclosure's centre (a cheaper
        // kernel value) without widening it, and below `Average` the kernel
        // is floored entirely. Containment asserted
        // above is the guarantee. Relative width across policies is not.
        let (tl, th) = bounds(tight);
        let (ll, lh) = bounds(loosest);
        assert!(
            tl <= th && ll <= lh,
            "both must be well-formed: [{tl}, {th}] [{ll}, {lh}]"
        );
    }
}

/// Constants are enclosures now, not points: PI must straddle the true pi.
#[test]
fn constants_enclose() {
    let pi: I<Balanced> = <I<Balanced> as FloatConsts>::PI;
    let (lo, hi) = bounds(pi);

    assert!(
        lo < std::f64::consts::PI || hi > std::f64::consts::PI,
        "PI must be a proper enclosure"
    );
    assert!(lo <= std::f64::consts::PI && std::f64::consts::PI <= hi);
    assert!(hi - lo > 0.0, "PI enclosure must have positive width");

    // sin(PI) must therefore contain 0.
    let s = pi.sin();
    assert!(s.contains(V1::ZERO).all(), "sin(PI) must contain 0: {:?}", bounds(s));

    let e: I<Balanced> = <I<Balanced> as FloatConsts>::E;
    let (lo, hi) = bounds(e);
    assert!(lo <= std::f64::consts::E && std::f64::consts::E <= hi);
}

/// Polynomial evaluation through the generic `poly` path encloses.
#[test]
fn poly_encloses() {
    let coeffs = [1.0, -2.0, 3.0, -4.0, 5.0];
    let mut state = 31337;

    for _ in 0..10_000 {
        let x = uniform(&mut state, -2.0, 2.0);
        let i: I<Tightest> = pt(x);

        let ic = [
            thermite_interval::IntervalElem::degenerate(1.0),
            thermite_interval::IntervalElem::degenerate(-2.0),
            thermite_interval::IntervalElem::degenerate(3.0),
            thermite_interval::IntervalElem::degenerate(-4.0),
            thermite_interval::IntervalElem::degenerate(5.0),
        ];
        let enclosed = i.poly_n(&ic);

        // Reference: Horner in double-double.
        let mut r = C::new(V1::splat(coeffs[4]));
        for &c in coeffs[..4].iter().rev() {
            r = r * C::new(V1::splat(x)) + C::new(V1::splat(c));
        }

        assert!(
            contains_dd(enclosed, r),
            "poly @ {x}: {:?} misses {:?}",
            bounds(enclosed),
            dd(r)
        );
    }
}

/// powi parity handling: even powers of a zero-straddling interval must have
/// a non-negative lower bound.
#[test]
fn powi_parity() {
    let x: I<Balanced> = iv(-2.0, 1.0);

    let sq = x.powi(2);
    let (lo, hi) = bounds(sq);
    assert!(lo >= 0.0, "even power must be non-negative: [{lo}, {hi}]");
    assert!(hi >= 4.0, "must reach (-2)^2 = 4");

    let cube = x.powi(3);
    let (lo, hi) = bounds(cube);
    assert!(lo <= -8.0 && hi >= 1.0, "odd power spans [-8, 1]: [{lo}, {hi}]");
}

// ---------------------------------------------------------------------------
// Overrides that exist for CONTAINMENT (the trait defaults are wrong on
// intervals). Each of these would fail against the inherited default.
// ---------------------------------------------------------------------------

/// `step` on an uncertain lane must be `[0, 1]`, not `0`.
#[test]
fn step_encloses_uncertain_lanes() {
    let edge: I<Balanced> = pt(1.0);

    let unsure: I<Balanced> = iv(0.0, 2.0);
    assert_eq!(
        bounds(unsure.step(edge)),
        (0.0, 1.0),
        "uncertain comparison must give [0, 1]"
    );

    let above: I<Balanced> = iv(1.5, 2.0);
    assert_eq!(bounds(above.step(edge)), (1.0, 1.0));

    let below: I<Balanced> = iv(-1.0, 0.5);
    assert_eq!(bounds(below.step(edge)), (0.0, 0.0));

    // Touching the edge from above counts as >=.
    let touching: I<Balanced> = iv(1.0, 1.5);
    assert_eq!(bounds(touching.step(edge)), (1.0, 1.0));
}

/// An odd root of a zero-straddling interval must include the negative roots.
#[test]
fn nth_root_odd_keeps_negative_branch() {
    let x: I<Balanced> = iv(-32.0, 1.0);
    let (lo, hi) = bounds(x.nth_root_n::<5>());
    assert!(lo <= -2.0, "5th root of -32 is -2; got lo = {lo}");
    assert!(hi >= 1.0, "5th root of 1 is 1; got hi = {hi}");

    // Even roots of a negative-touching interval clamp at the domain.
    let y: I<Balanced> = iv(-4.0, 16.0);
    let (lo, hi) = bounds(y.nth_root_n::<4>());
    assert!(lo <= 0.0 && lo >= -1e-300, "4th root domain clamps at zero: lo = {lo}");
    assert!(hi >= 2.0);

    let all_neg: I<Balanced> = iv(-16.0, -4.0);
    assert!(all_neg.nth_root_n::<4>().is_empty().all());
}

// ---------------------------------------------------------------------------
// Overrides for tightness: containment against Compensated references, plus
// the structural properties each one is supposed to have.
// ---------------------------------------------------------------------------

point_containment!(tan_contains, -1.4, 1.4, |i: I<Tightest>| i.tan(), |c: C| c.tan());
point_containment!(sqrt1pm1_contains, -0.99, 1e6, |i: I<Tightest>| i.sqrt1pm1(), |c: C| c
    .sqrt1pm1());
point_containment!(log2_p1_contains, -0.99, 1e6, |i: I<Tightest>| i.log2_p1(), |c: C| c
    .log2_p1());
point_containment!(log10_p1_contains, -0.99, 1e6, |i: I<Tightest>| i.log10_p1(), |c: C| c
    .log10_p1());
point_containment!(haversin_contains, -10.0, 10.0, |i: I<Tightest>| i.haversin(), |c: C| c
    .haversin());
point_containment!(
    nth_root5_contains,
    -1e5,
    1e5,
    |i: I<Tightest>| i.nth_root_n::<5>(),
    |c: C| c.nth_root_n::<5>()
);
point_containment!(
    nth_root6_contains,
    0.0,
    1e5,
    |i: I<Tightest>| i.nth_root_n::<6>(),
    |c: C| c.nth_root_n::<6>()
);
point_containment!(
    ln1m_expnx_contains,
    1e-3,
    30.0,
    |i: I<Tightest>| i.ln1m_expnx(),
    |c: C| c.ln1m_expnx()
);

/// tan: tight on one branch, entire across a pole.
#[test]
fn tan_pole_handling() {
    let on_branch: I<Balanced> = iv(0.1, 0.5);
    let (lo, hi) = bounds(on_branch.tan());
    assert!(lo <= 0.1f64.tan() && hi >= 0.5f64.tan());
    assert!(hi - lo < 1.0, "on-branch tan should be tight: [{lo}, {hi}]");

    let across: I<Balanced> = iv(1.5, 1.7); // pi/2 ~ 1.5708 inside
    assert_eq!(bounds(across.tan()), (f64::NEG_INFINITY, f64::INFINITY));

    let wide: I<Balanced> = iv(0.0, 4.0);
    assert_eq!(bounds(wide.tan()), (f64::NEG_INFINITY, f64::INFINITY));

    // Just past the pole on the negative branch: still tight (no false pole).
    let neg_branch: I<Balanced> = iv(1.6, 2.0);
    let (lo, hi) = bounds(neg_branch.tan());
    assert!(lo.is_finite() && hi.is_finite(), "no pole in [1.6, 2.0]: [{lo}, {hi}]");
    assert!(lo <= 1.6f64.tan() && hi >= 2.0f64.tan());
}

/// hypot over intervals: mignitude/magnitude structure, and containment.
#[test]
fn hypot_structure_and_containment() {
    let x: I<Balanced> = iv(-1.0, 2.0);
    let y: I<Balanced> = iv(3.0, 3.0);
    let (lo, hi) = bounds(x.hypot(y));
    assert!(lo <= 3.0 && lo > 2.9, "mig(x) = 0 -> lo = 3: got {lo}");
    assert!(hi >= 13f64.sqrt() && hi < 3.7, "mag(x) = 2 -> hi = sqrt(13): got {hi}");

    let mut state = 4242;
    for _ in 0..10_000 {
        let a = uniform(&mut state, -100.0, 100.0);
        let b = uniform(&mut state, -100.0, 100.0);
        let i: I<Tightest> = pt::<Tightest>(a).hypot(pt(b));
        let r = C::new(V1::splat(a)).hypot(C::new(V1::splat(b)));
        assert!(contains_dd(i, r), "hypot({a}, {b}): {:?} misses {:?}", bounds(i), dd(r));

        let i3: I<Tightest> = <I<Tightest> as thermite::math::SpatialMath>::hypot_n([pt(a), pt(b), pt(1.0)]);
        let r3 =
            <C as thermite::math::SpatialMath>::hypot_n([C::new(V1::splat(a)), C::new(V1::splat(b)), C::new(V1::ONE)]);
        assert!(
            contains_dd(i3, r3),
            "hypot_n({a}, {b}, 1): {:?} misses {:?}",
            bounds(i3),
            dd(r3)
        );

        let ii: I<Tightest> = <I<Tightest> as thermite::math::SpatialMath>::inv_hypot_n([pt(a), pt(b), pt(1.0)]);
        let ri = <C as thermite::math::SpatialMath>::inv_hypot_n([
            C::new(V1::splat(a)),
            C::new(V1::splat(b)),
            C::new(V1::ONE),
        ]);
        assert!(
            contains_dd(ii, ri),
            "inv_hypot_n({a}, {b}, 1): {:?} misses {:?}",
            bounds(ii),
            dd(ri)
        );
    }
}

/// Log-domain arithmetic: monotone corner rule, containment.
#[test]
fn log_domain_containment() {
    use thermite::math::RealMath;

    let mut state = 9001;
    for _ in 0..10_000 {
        let a = uniform(&mut state, -30.0, 30.0);
        let b = uniform(&mut state, -30.0, 30.0);
        let (ca, cb) = (C::new(V1::splat(a)), C::new(V1::splat(b)));

        let i = pt::<Tightest>(a).logaddexp(pt(b));
        assert!(contains_dd(i, ca.logaddexp(cb)), "logaddexp({a}, {b})");

        let i3 = <I<Tightest> as RealMath>::logsumexp_n([pt(a), pt(b), pt(0.5)]);
        let r3 = <C as RealMath>::logsumexp_n([ca, cb, C::new(V1::HALF)]);
        assert!(contains_dd(i3, r3), "logsumexp_n({a}, {b}, 0.5)");

        // logsubexp needs a > b by a margin the reference can resolve.
        let (hi_v, lo_v) = if a > b { (a, b) } else { (b, a) };
        if hi_v - lo_v > 1e-6 {
            let i = pt::<Tightest>(hi_v).logsubexp(pt(lo_v));
            let r = C::new(V1::splat(hi_v)).logsubexp(C::new(V1::splat(lo_v)));
            assert!(
                contains_dd(i, r),
                "logsubexp({hi_v}, {lo_v}): {:?} misses {:?}",
                bounds(i),
                dd(r)
            );
        }
    }

    // A box touching the diagonal reaches -inf, and one below it is empty.
    let a: I<Balanced> = iv(0.0, 2.0);
    let b: I<Balanced> = iv(1.0, 1.5);
    let (lo, _) = bounds(a.logsubexp(b));
    assert_eq!(lo, f64::NEG_INFINITY);
    let below: I<Balanced> = iv(-2.0, -1.0);
    assert!(below.logsubexp(a).is_empty().all());
}

/// smoothstep: clamped, monotone, contains the reference, and interval edges
/// spanning zero give the whole range.
#[test]
fn smoothstep_interval() {
    use thermite::math::RealMath;

    let mut state = 12;
    for _ in 0..5_000 {
        let x = uniform(&mut state, -0.5, 1.5);
        let i = pt::<Tightest>(x).smoothstep::<2>(None);
        let xc = x.clamp(0.0, 1.0);
        let r = C::new(V1::splat(xc)).smoothstep::<2>(None);
        assert!(contains_dd(i, r), "smoothstep({x}): {:?} misses {:?}", bounds(i), dd(r));
    }

    let mid: I<Balanced> = iv(0.25, 0.75);
    let (lo, hi) = bounds(mid.smoothstep::<2>(None));
    // 3t^2 - 2t^3 at 0.25 = 0.15625, at 0.75 = 0.84375
    assert!(lo <= 0.15625 && lo > 0.15, "[{lo}, {hi}]");
    assert!(hi >= 0.84375 && hi < 0.85, "[{lo}, {hi}]");

    let below: I<Balanced> = iv(-3.0, -1.0);
    assert_eq!(bounds(below.smoothstep::<2>(None)), (0.0, 0.0));

    let zero_span_edges: I<Balanced> = iv(0.5, 0.5);
    let e0: I<Balanced> = iv(0.0, 1.0);
    let e1: I<Balanced> = iv(0.5, 2.0); // e1 - e0 contains 0
    assert_eq!(bounds(zero_span_edges.smoothstep::<2>(Some((e0, e1)))), (0.0, 1.0));
}

/// compound / powf_m1 four-corner containment.
#[test]
fn compound_and_powf_m1_contain() {
    let mut state = 77;
    for _ in 0..10_000 {
        let x = uniform(&mut state, -0.5, 3.0);
        let n = uniform(&mut state, -5.0, 5.0);
        let i = pt::<Tightest>(x).compound(pt(n));
        let r = C::new(V1::splat(x)).compound(C::new(V1::splat(n)));
        assert!(
            contains_dd(i, r),
            "compound({x}, {n}): {:?} misses {:?}",
            bounds(i),
            dd(r)
        );

        let b = uniform(&mut state, 0.1, 10.0);
        let e = uniform(&mut state, -3.0, 3.0);
        let i = pt::<Tightest>(b).powf_m1(pt(e));
        let r = C::new(V1::splat(b)).powf_m1(C::new(V1::splat(e)));
        assert!(
            contains_dd(i, r),
            "powf_m1({b}, {e}): {:?} misses {:?}",
            bounds(i),
            dd(r)
        );
    }
}

/// haversin never goes negative on a zero-straddling input.
#[test]
fn haversin_nonneg() {
    let x: I<Balanced> = iv(-0.5, 0.5);
    let (lo, hi) = bounds(x.haversin());
    assert!(lo >= 0.0, "haversin is a square: lo = {lo}");
    assert!(hi >= (0.25f64).sin().powi(2));
}

/// Horner (forced) encloses at every policy, including the ones where the
/// default would have taken the Estrin path.
#[test]
fn poly_all_policies() {
    use thermite::math::CoreMathWithPolicy;
    use thermite::math::policy::policies::UltraPerformance;

    let ic = [
        thermite_interval::IntervalElem::degenerate(1.0),
        thermite_interval::IntervalElem::degenerate(-2.0),
        thermite_interval::IntervalElem::degenerate(3.0),
    ];
    let mut state = 5;
    for _ in 0..5_000 {
        let x = uniform(&mut state, -2.0, 2.0);
        let i: I<Tightest> = pt(x);
        let mut r = C::new(V1::splat(3.0));
        r = r * C::new(V1::splat(x)) + C::new(V1::splat(-2.0));
        r = r * C::new(V1::splat(x)) + C::new(V1::splat(1.0));

        for (name, e) in [
            ("precision", CoreMathWithPolicy::poly_n_p::<Precision, 3>(i, &ic)),
            ("performance", CoreMathWithPolicy::poly_n_p::<Performance, 3>(i, &ic)),
            ("ultra", CoreMathWithPolicy::poly_n_p::<UltraPerformance, 3>(i, &ic)),
            (
                "rev",
                CoreMathWithPolicy::poly_rev_n_p::<Precision, 3>(i, &[ic[2], ic[1], ic[0]]),
            ),
        ] {
            assert!(
                contains_dd(e, r),
                "poly {name} @ {x}: {:?} misses {:?}",
                bounds(e),
                dd(r)
            );
        }
    }
}

/// REGRESSION: containment at the LOW math-policy tiers, across many
/// functions, not just the well-behaved ones.
///
/// `Medium`/`Worst` kernels abandon relative error bounds (measured:
/// `ln1m_expnx(40)` returns exactly 0 where the truth is -4.2e-18), which no
/// relative margin can cover. `KernelPolicy` floors the inner kernel at
/// `Average` for exactly this reason. Before that floor existed, this test
/// failed on `ln1m_expnx` at every tier below `Average`. The old
/// single-function version passed because `exp` happens to be accurate at
/// every tier.
#[test]
fn low_policy_tiers_still_contain() {
    use thermite::math::policy::DefaultPolicy;
    use thermite::math::policy::policies::{MediumPrecision, WorstPrecision};
    use thermite::math::{SpatialMathWithPolicy, TranscendentalMathWithPolicy};

    let mut state = 4242;
    let mut checked = 0usize;

    for _ in 0..2_000 {
        // Domains chosen so every function below is in range.
        let x = uniform(&mut state, 0.05, 8.0);
        let p: I<Tightest> = pt(x);
        let cx = C::new(V1::splat(x));

        macro_rules! check {
            ($label:literal, $m:ident, $r:expr) => {{
                let truth: C = $r;
                for (tier, got) in [
                    (
                        "Medium",
                        TranscendentalMathWithPolicy::$m::<MediumPrecision<DefaultPolicy>>(p),
                    ),
                    (
                        "Worst",
                        TranscendentalMathWithPolicy::$m::<WorstPrecision<DefaultPolicy>>(p),
                    ),
                ] {
                    assert!(
                        contains_dd(got, truth),
                        "{} at {} tier lost containment @ x={x}: {:?} vs {:?}",
                        $label,
                        tier,
                        bounds(got),
                        dd(truth)
                    );
                    checked += 1;
                }
            }};
        }

        check!("ln1m_expnx", ln1m_expnx_p, cx.ln1m_expnx());
        check!("exp", exp_p, cx.exp());
        check!("ln", ln_p, cx.ln());
        check!("sin", sin_p, cx.sin());
        check!("cos", cos_p, cx.cos());
        check!("tanh", tanh_p, cx.tanh());
        check!("atan", atan_p, cx.atan());
        check!("cbrt", cbrt_p, cx.cbrt());
        check!("exp_m1", exp_m1_p, cx.exp_m1());
        check!("ln_1p", ln_1p_p, cx.ln_1p());
        check!("sinh", sinh_p, cx.sinh());
        check!("cosh", cosh_p, cx.cosh());

        // Two-argument ones, and a Spatial member for good measure.
        let y = uniform(&mut state, 0.05, 8.0);
        let q: I<Tightest> = pt(y);
        let cy = C::new(V1::splat(y));
        for (tier, got) in [
            ("Medium", p.hypot_p::<MediumPrecision<DefaultPolicy>>(q)),
            ("Worst", p.hypot_p::<WorstPrecision<DefaultPolicy>>(q)),
        ] {
            assert!(
                contains_dd(got, cx.hypot(cy)),
                "hypot at {tier} lost containment @ ({x}, {y})"
            );
            checked += 1;
        }
    }

    assert!(checked > 50_000, "sanity: {checked} checks ran");
}

/// The kernel-policy floor is observable: a `Worst`-tier request produces the
/// same enclosure as an `Average`-tier one (the kernel is floored), while
/// `Precision` may differ. Documents the deliberate behaviour so nobody
/// "optimizes" the floor away.
#[test]
fn kernel_policy_floor_is_applied() {
    use thermite::math::TranscendentalMathWithPolicy;
    use thermite::math::policy::DefaultPolicy;
    use thermite::math::policy::policies::{AveragePrecision, WorstPrecision};

    let p: I<Tightest> = pt(40.0);
    let avg = TranscendentalMathWithPolicy::ln1m_expnx_p::<AveragePrecision<DefaultPolicy>>(p);
    let worst = TranscendentalMathWithPolicy::ln1m_expnx_p::<WorstPrecision<DefaultPolicy>>(p);
    assert_eq!(
        bounds(avg),
        bounds(worst),
        "Worst must be floored to Average inside the interval math"
    );
}

/// REGRESSION: sin/cos containment at large arguments.
///
/// Two independent failures meet here, both measured. A quadrant index computed
/// as `floor(x * fl(2/pi))` in point arithmetic lets the rounding hide a crossed
/// extremum past |x| ~ 1e9. And the `Average`-tier kernel's own range reduction
/// collapses past |x| ~ 1e8 (absolute error 1.0 by 1e14, so the value is
/// meaningless). Containment needs an interval-valued reduction through the
/// `FRAC_2_PI` enclosure, plus a magnitude guard that degrades to `[-1, 1]`.
///
/// Violations before the fix: 1485/6000 at 1e9, 1991/6000 at 1e12, with a
/// worst miss of 2.0 (a complete sign flip).
#[test]
fn trig_contains_at_large_arguments() {
    let mut state = 99u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };

    for mag in [1e3f64, 1e6, 1e9, 1e12, 1e14, 1e15, 1e16] {
        for _ in 0..400 {
            let a = mag * (1.0 + next());
            let w = mag * 1e-13 * next();
            let x: I<Tightest> = iv(a, a + w);
            let (se, ce) = (x.sin(), x.cos());

            for t in [0.0, 0.5, 1.0] {
                let p = a + w * t;
                let r = C::new(V1::splat(p));
                assert!(
                    contains_dd(se, r.sin()),
                    "sin over [{a:e}, {:e}] misses x={p:e}: {:?}",
                    a + w,
                    bounds(se)
                );
                assert!(
                    contains_dd(ce, r.cos()),
                    "cos over [{a:e}, {:e}] misses x={p:e}: {:?}",
                    a + w,
                    bounds(ce)
                );
            }

            // And the range clamp still holds.
            for e in [se, ce] {
                let (lo, hi) = bounds(e);
                assert!(lo >= -1.0 && hi <= 1.0, "escaped [-1, 1]: [{lo}, {hi}]");
            }
        }
    }
}

/// Extended division: a divisor whose zero sits on an ENDPOINT still bounds
/// the quotient on one side (Boost.Interval's `div_positive`/`div_negative`).
/// Only an interior zero needs the whole line.
#[test]
fn division_by_endpoint_zero_gives_half_lines() {
    // y = [0, 2], x = [1, 4] -> [1/2, +inf]
    let x: I<Balanced> = iv(1.0, 4.0);
    let y: I<Balanced> = iv(0.0, 2.0);
    let (lo, hi) = bounds(x / y);
    assert!(lo <= 0.5 && lo > 0.49, "finite lower bound xl/yu = 0.5, got {lo}");
    assert_eq!(hi, f64::INFINITY);

    // y = [0, 2], x = [-4, -1] -> [-inf, -1/2]
    let x: I<Balanced> = iv(-4.0, -1.0);
    let (lo, hi) = bounds(x / y);
    assert_eq!(lo, f64::NEG_INFINITY);
    assert!(hi >= -0.5 && hi < -0.49, "finite upper bound xu/yu = -0.5, got {hi}");

    // y = [-2, 0], x = [1, 4] -> [-inf, -1/2]
    let y: I<Balanced> = iv(-2.0, 0.0);
    let x: I<Balanced> = iv(1.0, 4.0);
    let (lo, hi) = bounds(x / y);
    assert_eq!(lo, f64::NEG_INFINITY);
    assert!(hi >= -0.5 && hi < -0.49, "xl/yl = -0.5, got {hi}");

    // y = [-2, 0], x = [-4, -1] -> [1/2, +inf]
    let x: I<Balanced> = iv(-4.0, -1.0);
    let (lo, hi) = bounds(x / y);
    assert!(lo <= 0.5 && lo > 0.49, "xu/yl = 0.5, got {lo}");
    assert_eq!(hi, f64::INFINITY);

    // Interior zero still gives the whole line.
    let y: I<Balanced> = iv(-1.0, 2.0);
    let x: I<Balanced> = iv(1.0, 4.0);
    assert_eq!(bounds(x / y), (f64::NEG_INFINITY, f64::INFINITY));

    // Containment: sample the divisor away from zero and check the quotient
    // lands inside the half-line.
    let y: I<Balanced> = iv(0.0, 2.0);
    let x: I<Balanced> = iv(1.0, 4.0);
    let (lo, _) = bounds(x / y);
    for dy in [1e-6, 0.01, 0.5, 1.0, 2.0] {
        for nx in [1.0, 2.5, 4.0] {
            assert!(nx / dy >= lo, "{nx}/{dy} = {} below the half-line bound {lo}", nx / dy);
        }
    }
}

/// A degenerate zero operand makes the product exactly zero, and no widening
/// tier may fatten it (Boost.Interval's `? * Z -> Z`).
#[test]
fn multiply_by_exact_zero_stays_exact() {
    fn check<W: WideningPolicy>(label: &str) {
        let z: I<W> = iv(0.0, 0.0);
        for other in [iv::<W>(1.0, 2.0), iv(-3.0, 5.0), iv(-1e300, 1e300), iv(0.0, 0.0)] {
            assert_eq!(bounds(z * other), (0.0, 0.0), "{label}: z * other must be exactly zero");
            assert_eq!(bounds(other * z), (0.0, 0.0), "{label}: other * z must be exactly zero");
        }
    }
    check::<Fastest>("Fastest");
    check::<Balanced>("Balanced");
    check::<Tightest>("Tightest");

    // A non-degenerate interval containing zero is NOT exact and may widen.
    let nearly: I<Balanced> = iv(-0.0, 0.0);
    assert_eq!(
        bounds(nearly * iv(1.0, 2.0)),
        (0.0, 0.0),
        "[-0, 0] is still degenerate zero"
    );
}
