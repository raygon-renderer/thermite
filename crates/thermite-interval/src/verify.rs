//! Verified root-finding: the Krawczyk operator over SIMD lanes of boxes.
//!
//! Requires the `dual` feature. `Dual<Interval<V, W>, 1>` evaluates a
//! function `f` and encloses its derivative `F'(X)` in one sweep. The chain
//! rule runs over interval arithmetic, so every derivative is itself a
//! rigorous enclosure. That is exactly what the Krawczyk operator needs:
//!
//! ```text
//! K(X) = m - C f(m) + (1 - C F'(X)) (X - m),    m = mid(X), C ~ 1/f'(m)
//! ```
//!
//! - `K(X)` strictly inside `X`   =>  `X` contains exactly one root (proof).
//! - `K(X)` disjoint from `X`     =>  `X` contains no root (proof).
//! - otherwise                    =>  shrink to `X ∩ K(X)` and bisect.
//!
//! Every lane carries its own box, so one call yields `LANES` independent
//! existence certificates. `C` is any real number (a degenerate interval).
//! Its choice affects how fast `K` contracts, never the validity of the
//! verdicts, which rest on containment alone.
//!
//! `DualValue for Interval` lives here (not in thermite-dual) because this
//! module needs `Dual`, so thermite-interval must depend on thermite-dual.
//! Hosting the impl in thermite-dual too would need the reverse dependency.

use thermite::prelude::*;
use thermite_dual::{Dual, DualValue};

use crate::consts::BoundedFloatConsts;
use crate::element::{IntervalElem, ScalarFloat};
use crate::math::IntervalMathVector;
use crate::widen::WideningPolicy;
use crate::{Interval, IntervalFloatVector};

impl<V: IntervalFloatVector + BoundedFloatConsts<V>, W: WideningPolicy> DualValue for Interval<V, W> {
    const VAL_ZERO: Self = <Self as NumericVector>::ZERO;
    const VAL_ONE: Self = <Self as NumericVector>::ONE;

    #[inline(always)]
    fn val_trunc(self) -> Self {
        FloatVector::trunc(self)
    }
}

impl<E: ScalarFloat> DualValue for IntervalElem<E> {
    const VAL_ZERO: Self = <Self as thermite::element::Element>::ZERO;
    const VAL_ONE: Self = <Self as thermite::element::Element>::ONE;

    #[inline(always)]
    fn val_trunc(self) -> Self {
        thermite::element::FloatElement::trunc(self)
    }
}

/// Per-lane verdicts of one Krawczyk step.
#[derive(Clone, Copy, Debug)]
pub struct Existence<M> {
    /// `K(X)` lies strictly inside `X`: exactly one root in `X`, proven.
    pub unique: M,
    /// `K(X)` does not meet `X`: no root in `X`, proven.
    pub excluded: M,
}

/// The value-and-derivative type the Krawczyk step evaluates `f` on.
pub type DualInterval<V, W> = Dual<Interval<V, W>, 1>;

/// One Krawczyk step over a lane-vector of boxes.
///
/// `f` is evaluated twice: on the seeded box (`Dual::variable(X)`) for the
/// derivative enclosure `F'(X)`, and on the degenerate midpoint
/// (`Dual::constant(m)`) for the enclosure of `f(m)`. Returns the contracted
/// boxes `X ∩ K(X)` (empty where excluded) and the per-lane verdicts.
///
/// Empty input lanes stay empty and are reported as `excluded`.
#[inline(always)]
pub fn krawczyk_step<V, W, F>(f: F, x: Interval<V, W>) -> (Interval<V, W>, Existence<V::Mask>)
where
    V: IntervalMathVector,
    W: WideningPolicy,
    F: Fn(DualInterval<V, W>) -> DualInterval<V, W>,
{
    let m_pt = x.midpoint();
    let m = Interval::<V, W>::degenerate(m_pt);

    // Derivative enclosure over the box, and f at the midpoint (enclosed).
    let dfx = f(Dual::variable(x, 0)).dual[0];
    let fm = f(Dual::constant(m)).re;

    // Preconditioner: 1 / mid(F'(X)) as a point. Where the derivative
    // enclosure is centred on zero (or non-finite), fall back to C = 1: still
    // valid, merely a poor contraction, and the step reports "uncertain".
    let dmid = dfx.midpoint();
    let usable = dmid.is_finite() & !dmid.is_zero();
    let c_pt = usable.select(V::ONE / dmid, V::ONE);
    let c = Interval::<V, W>::degenerate(c_pt);

    let one = Interval::<V, W>::degenerate(V::ONE);
    let k = m - c * fm + (one - c * dfx) * (x - m);

    // Strict interior: K.lo > X.lo and K.hi < X.hi (Krawczyk's uniqueness
    // condition), on non-empty lanes.
    let empty_in = x.is_empty();
    let nonempty = !empty_in & !k.is_empty();
    let unique = nonempty & k.lo().cmp_gt(x.lo()) & k.hi().cmp_lt(x.hi());
    let excluded = empty_in | k.intersect(x).is_empty();

    // An empty input lane has a NaN midpoint, which poisons K. Force those
    // lanes back to the canonical empty encoding so callers (and iterated
    // refinement) see a clean empty rather than a NaN box.
    let r = x.intersect(k);
    let r = Interval::<V, W>::from_bounds_unchecked(
        empty_in.select(V::INFINITY, r.lo()),
        empty_in.select(V::NEG_INFINITY, r.hi()),
    );

    (r, Existence { unique, excluded })
}

/// A certified root enclosure produced by [`certify_roots`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CertifiedRoot<E> {
    /// The box. Exactly one root of `f` lies inside (proof by Krawczyk).
    pub lo: E,
    pub hi: E,
}

/// Outcome of a subdivision run over `[a, b]`.
#[derive(Clone, Debug, Default)]
pub struct Certification<E> {
    /// Boxes proven to contain exactly one root each.
    pub roots: alloc::vec::Vec<CertifiedRoot<E>>,
    /// Boxes below the width tolerance that could be neither certified nor
    /// excluded (multiple/tangent roots, or a root exactly on a subdivision
    /// boundary). Not proofs of anything, but reported so nothing is lost.
    pub unresolved: alloc::vec::Vec<CertifiedRoot<E>>,
    /// Krawczyk steps executed (each over one full lane-vector of boxes).
    pub steps: usize,
    /// Boxes examined in total.
    pub boxes: usize,
}

/// Certifies every root of `f` on `[a, b]` by subdivision, `V::LANES` boxes per
/// Krawczyk step, then refines each certified box to width `tol` by iterating
/// the operator (which contracts quadratically and preserves the proof).
///
/// Boxes proven root-free are dropped. Boxes proven to hold exactly one root
/// are collected and refined in lane-wide batches. The rest are contracted,
/// bisected, and requeued until narrower than `tol`. Requires `alloc` for the
/// work list, though the arithmetic itself is `no_std`.
///
/// Bisection splits at ratio 33/64 rather than 1/2 (still an exact binary
/// fraction, so no rounding), because a root sitting exactly on a cut can
/// never be strictly interior to either child and would drop to `unresolved`.
/// Round-number roots on round-number search intervals hit the midpoint
/// constantly. Multiple (tangent) roots still land in `unresolved`: Krawczyk
/// cannot certify uniqueness at a tangency, which is the honest answer.
pub fn certify_roots<V, W, F>(f: F, a: V::Element, b: V::Element, tol: V::Element, max_boxes: usize) -> Certification<V::Element>
where
    V: IntervalMathVector,
    V::Element: PartialOrd + Copy,
    W: WideningPolicy,
    F: Fn(DualInterval<V, W>) -> DualInterval<V, W>,
{
    use alloc::vec::Vec;

    let lanes = V::LANES;
    let zero = <V::Element as thermite::element::Element>::ZERO;
    let split = <V::Element as thermite::element::FloatElement>::from_ratio(33, 64);

    let mut work: Vec<(V::Element, V::Element)> = alloc::vec![(a, b)];
    let mut certified: Vec<(V::Element, V::Element)> = Vec::new();
    let mut out = Certification::default();

    // Packs up to `lanes` boxes into one interval vector (empty lanes pad).
    let pack = |src: &mut Vec<(V::Element, V::Element)>| -> (Interval<V, W>, usize) {
        let mut lo = Interval::<V, W>::empty().lo();
        let mut hi = Interval::<V, W>::empty().hi();
        let mut n = 0;
        while n < lanes {
            let Some((l, h)) = src.pop() else { break };
            lo = lo.insertv(n, l);
            hi = hi.insertv(n, h);
            n += 1;
        }
        (Interval::<V, W>::from_bounds_unchecked(lo, hi), n)
    };

    // --- phase 1: subdivide until every box is certified, excluded, or tiny --
    while !work.is_empty() && out.boxes < max_boxes {
        let (x, n) = pack(&mut work);
        out.boxes += n;
        out.steps += 1;

        let (contracted, verdict) = krawczyk_step(&f, x);
        // Masks have no per-lane read, so materialize them as 0/1 vectors.
        let excluded_v = verdict.excluded.select(V::ONE, V::ZERO);
        let unique_v = verdict.unique.select(V::ONE, V::ZERO);

        for i in 0..n {
            let (xl, xh) = (x.lo().extractv(i), x.hi().extractv(i));
            let (cl, ch) = (contracted.lo().extractv(i), contracted.hi().extractv(i));

            if excluded_v.extractv(i) != zero {
                continue;
            }
            if unique_v.extractv(i) != zero {
                certified.push((cl, ch));
                continue;
            }
            // Uncertain: bisect the contracted box (the original if the
            // contraction did not narrow it).
            let (l, h) = if cl <= ch && (ch - cl) < (xh - xl) { (cl, ch) } else { (xl, xh) };
            let width = h - l;
            if width < tol {
                out.unresolved.push(CertifiedRoot { lo: l, hi: h });
                continue;
            }
            let m = l + width * split;
            work.push((l, m));
            work.push((m, h));
        }
    }

    // --- phase 2: refine certified boxes to `tol` by iterating Krawczyk -----
    // Each further step on a certified box keeps K(X) inside X (the root is
    // unique and the operator contracts toward it), so the proof survives and
    // the width shrinks quadratically. Bounded to 64 iterations as a
    // backstop, since convergence to a few ulps takes single digits.
    while !certified.is_empty() {
        let (mut x, n) = pack(&mut certified);
        let mut iters = 0;
        loop {
            let (next, _) = krawczyk_step(&f, x);
            iters += 1;
            out.steps += 1;
            // Stop when every live lane is under tolerance or nothing moved.
            let width = next.width();
            let done = width.cmp_lt(V::splat(tol)) | next.is_empty();
            let stalled = next.lo().cmp_eq(x.lo()) & next.hi().cmp_eq(x.hi());
            x = next;
            if (done | stalled).all() || iters >= 64 {
                break;
            }
        }
        for i in 0..n {
            out.roots.push(CertifiedRoot { lo: x.lo().extractv(i), hi: x.hi().extractv(i) });
        }
    }

    out
}
