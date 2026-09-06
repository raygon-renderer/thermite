//! Rigorous SIMD interval arithmetic built on Thermite.
//!
//! An [`Interval<V, W>`] carries a closed interval `[lo, hi]` per lane, and
//! every operation returns an enclosure of the exact real result
//! (_containment_). Because `Interval` implements the same vector traits as
//! every other Thermite composite, generic kernels written against
//! `FloatVector` bounds compute verified enclosures with no changes.
//!
//! # Two orthogonal policies
//!
//! - The **widening policy `W`** ([`Fastest`] / [`Balanced`] / [`Tightest`])
//!   lives on the type and governs the _interval bookkeeping_, meaning how
//!   outward rounding is performed on every operation, including the
//!   policy-less surfaces (`+`, `sqrt`, `mul_adde`). Mixing tiers is a type
//!   error, so convert explicitly with [`with_widening`](Interval::with_widening).
//! - Thermite's **math policy `P`** keeps its usual role on the (future)
//!   math-library methods: it tunes the _approximation algorithms_. A loose
//!   `sin_p::<Performance>` over `Tightest` intervals is a cheap value that is
//!   still honestly enclosed. The enclosure absorbs the algorithm's documented
//!   error bound while the endpoint arithmetic stays tight.
//!
//! Containment is never negotiable: neither policy may produce an interval
//! that fails to contain the true result.
//!
//! Outward rounding never touches the hardware rounding mode. It widens by
//! ulp-stepping or eps-scaling after nearest-rounded ops, or steps only where
//! an error-free transform proves rounding actually erred (the residual
//! strategy, which also keeps degenerate intervals degenerate through exact
//! operations).
//!
//! # Rigor
//!
//! - **Rigorous**: arithmetic (`+ - * /`), `sqrt`, `square`, `abs`,
//!   `min`/`max`, the set operations, and the [`FloatConsts`] constants.
//!   These use only correctly-rounded primitives, error-free transforms, and
//!   the [`BoundedFloatConsts`] enclosure table (`[next_down(fl(c)),
//!   next_up(fl(c))]`, which brackets the true constant since `fl(c)` is
//!   within half an ulp).
//! - **High-confidence, not yet certified**: the transcendental library.
//!   Endpoint values come from thermite's kernels, and the enclosure widens
//!   by a conservative per-policy algorithm-error margin
//!   (`math::algo_widen`). Those margins are engineering estimates over the
//!   kernels' documented accuracies, not proofs (see the PROVISIONAL note
//!   in [`math`]). Certified per-function bounds would need a Gappa-style
//!   proof pass.
//!
//! # Known limitations (early crate)
//!
//! - Several `GenericVector` plumbing methods (memory loads/stores, lookup)
//!   are `todo!()` until the interleaved memory layout is settled, so slice
//!   iteration does not work yet.
//! - Functions without a hand-written interval form (`tan`, `sin_pi`,
//!   `smoothstep_derivative`, `logsumexp_n`, ...) fall back to the trait defaults.
//!   Those compose out of the enclosing arithmetic, so they are _valid_ but
//!   can be very wide (the dependency problem). They are the next targets.
//! - `Interval` deliberately does not implement `FloatVectorWithBits` (no
//!   bit-level view of an interval) or the sort traits.

#![no_std]

#[cfg(feature = "dual")]
extern crate alloc;

use core::marker::PhantomData;

use thermite::element::FloatElement;
use thermite::math::PrimalProjection;
use thermite::prelude::*;
use thermite::vector::SwizzleVector;
use thermite_compensated::ScalarValue;

pub mod consts;
#[doc(hidden)]
pub mod consts_table;
pub mod element;
pub mod math;
pub(crate) mod ops;
pub(crate) mod round;
pub(crate) mod vector;
#[cfg(feature = "dual")]
#[cfg_attr(docsrs, doc(cfg(feature = "dual")))]
pub mod verify;
pub mod widen;

pub use consts::BoundedFloatConsts;
pub use element::{IntervalElem, ScalarFloat};
pub use widen::{Balanced, Fastest, Tightest, WideningPolicy, WideningTier};

/// Inner vector requirements for interval arithmetic: a real float vector
/// whose element supports scalar comparison (for the scalar element form),
/// carrying thermite-compensated's [`ScalarValue`] for the canonical
/// error-free transforms (`two_sum`, Veltkamp-split `two_prod`) that the
/// residual widening tier is built on.
pub trait IntervalFloatVector:
    FloatVector<Element: FloatElement + PartialOrd + num_traits::NumOps> + ScalarValue + CastVector<Self> + SwizzleVector
{
}
impl<V> IntervalFloatVector for V where
    V: FloatVector<Element: FloatElement + PartialOrd + num_traits::NumOps>
        + ScalarValue
        + CastVector<V>
        + SwizzleVector
{
}

/// A closed interval `[lo, hi]` per lane, widened per the policy `W`.
///
/// Invariant: `lo <= hi` in every lane, or the lane is _empty_, encoded as
/// `[+inf, -inf]` so that `hull`/`intersect` fall out of `min`/`max` with no
/// branches. Interval operations never produce NaN endpoints. Invalid inputs
/// yield empty lanes instead.
///
/// Fields are private so that arithmetic cannot break the invariant
/// silently. Construct with [`degenerate`](Self::degenerate),
/// [`bounds`](Self::bounds), or [`from_midrad`](Self::from_midrad), and read
/// with [`lo`](Self::lo)/[`hi`](Self::hi).
#[repr(C)]
pub struct Interval<V, W = Balanced> {
    pub(crate) lo: V,
    pub(crate) hi: V,
    pub(crate) _widen: PhantomData<W>,
}

// Manual impls: derives would demand the bounds on `W` through PhantomData in
// awkward ways. These are exactly the component impls.
impl<V: Clone, W> Clone for Interval<V, W> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self {
            lo: self.lo.clone(),
            hi: self.hi.clone(),
            _widen: PhantomData,
        }
    }
}
impl<V: Copy, W> Copy for Interval<V, W> {}

impl<V: core::fmt::Debug, W> core::fmt::Debug for Interval<V, W> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Interval")
            .field("lo", &self.lo)
            .field("hi", &self.hi)
            .finish()
    }
}

impl<V: Default, W> Default for Interval<V, W> {
    #[inline(always)]
    fn default() -> Self {
        Self {
            lo: V::default(),
            hi: V::default(),
            _widen: PhantomData,
        }
    }
}

/// Structural equality of the representation (both bounds bitwise-compare
/// equal per the inner vector's `PartialEq`). Set equality of empty lanes
/// with different encodings is NOT collapsed.
impl<V: PartialEq, W> PartialEq for Interval<V, W> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.lo == other.lo && self.hi == other.hi
    }
}

impl<V, W> Interval<V, W> {
    /// Constructs from raw bounds without checking `lo <= hi`.
    ///
    /// Not `unsafe` (no memory safety is at stake), but a violated invariant
    /// makes every subsequent containment claim meaningless.
    #[inline(always)]
    pub const fn from_bounds_unchecked(lo: V, hi: V) -> Self {
        Self {
            lo,
            hi,
            _widen: PhantomData,
        }
    }

    /// The lower bound of each lane.
    #[inline(always)]
    pub fn lo(self) -> V
    where
        V: Copy,
    {
        self.lo
    }

    /// The upper bound of each lane.
    #[inline(always)]
    pub fn hi(self) -> V
    where
        V: Copy,
    {
        self.hi
    }

    /// Reinterprets under a different widening policy. Free, since the
    /// representation is identical. The bounds are unchanged, so this is
    /// always containment-sound. Only _future_ operations widen differently.
    #[inline(always)]
    pub fn with_widening<W2>(self) -> Interval<V, W2> {
        Interval {
            lo: self.lo,
            hi: self.hi,
            _widen: PhantomData,
        }
    }
}

impl<V: IntervalFloatVector, W: WideningPolicy> Interval<V, W> {
    /// The degenerate interval `[v, v]`: an exactly-known value.
    #[inline(always)]
    pub fn degenerate(v: V) -> Self {
        Self::from_bounds_unchecked(v, v)
    }

    /// Constructs `[lo, hi]`, mapping lanes where `lo > hi` (or either bound
    /// is NaN) to empty.
    #[inline(always)]
    pub fn bounds(lo: V, hi: V) -> Self {
        let valid = lo.cmp_le(hi); // false for NaN in either bound
        Self::from_bounds_unchecked(valid.select(lo, V::INFINITY), valid.select(hi, V::NEG_INFINITY))
    }

    /// The whole real line, `[-inf, +inf]`, in every lane.
    #[inline(always)]
    pub fn entire() -> Self {
        Self::from_bounds_unchecked(V::NEG_INFINITY, V::INFINITY)
    }

    /// The empty interval in every lane.
    #[inline(always)]
    pub fn empty() -> Self {
        Self::from_bounds_unchecked(V::INFINITY, V::NEG_INFINITY)
    }

    /// `[mid - rad, mid + rad]`, outward-rounded so the true ball is enclosed.
    #[inline(always)]
    pub fn from_midrad(mid: V, rad: V) -> Self {
        Self::from_bounds_unchecked((mid - rad).next_down(), (mid + rad).next_up())
    }

    /// Per-lane mask of empty lanes.
    #[inline(always)]
    pub fn is_empty(self) -> V::Mask {
        self.lo.cmp_gt(self.hi)
    }

    /// `hi - lo`, rounded up (an upper bound on the true width). Empty lanes
    /// give `-inf`.
    #[inline(always)]
    pub fn width(self) -> V {
        (self.hi - self.lo).next_up()
    }

    /// An approximate midpoint. NOT guaranteed to lie inside for near-empty
    /// or infinite lanes. Use it for heuristics (subdivision pivots), never
    /// for containment arguments.
    #[inline(always)]
    pub fn midpoint(self) -> V {
        self.lo.mul_adde(V::HALF, self.hi * V::HALF)
    }

    /// An upper bound on the distance from [`midpoint`](Self::midpoint) to
    /// either endpoint.
    #[inline(always)]
    pub fn radius(self) -> V {
        let m = self.midpoint();
        (m - self.lo).max(self.hi - m).next_up()
    }

    /// Largest absolute value in the interval: `max(|lo|, |hi|)`.
    #[inline(always)]
    pub fn magnitude(self) -> V {
        self.lo.abs().max(self.hi.abs())
    }

    /// Smallest absolute value in the interval: 0 if the interval contains 0,
    /// else `min(|lo|, |hi|)`.
    #[inline(always)]
    pub fn mignitude(self) -> V {
        let contains_zero = self.lo.cmp_le(V::ZERO) & self.hi.cmp_ge(V::ZERO);
        contains_zero.select(V::ZERO, self.lo.abs().min(self.hi.abs()))
    }

    /// Per-lane mask: does this interval contain the point `v`?
    #[inline(always)]
    pub fn contains(self, v: V) -> V::Mask {
        self.lo.cmp_le(v) & v.cmp_le(self.hi)
    }

    /// Per-lane mask: is `self` a subset of `other`? Empty lanes of `self`
    /// are subsets of everything.
    #[inline(always)]
    pub fn subset_of(self, other: Self) -> V::Mask {
        (other.lo.cmp_le(self.lo) & self.hi.cmp_le(other.hi)) | self.is_empty()
    }

    /// Set intersection. Lanes with no overlap come out empty (the `[+inf,
    /// -inf]` encoding is exactly what max/min produce there).
    #[inline(always)]
    pub fn intersect(self, other: Self) -> Self {
        Self::from_bounds_unchecked(self.lo.max(other.lo), self.hi.min(other.hi))
    }

    /// Interval hull (smallest interval containing both). Empty lanes are
    /// identity elements, again for free from min/max.
    #[inline(always)]
    pub fn hull(self, other: Self) -> Self {
        Self::from_bounds_unchecked(self.lo.min(other.lo), self.hi.max(other.hi))
    }

    /// Splits at the midpoint: `([lo, m], [m, hi])`, for subdivision drivers.
    #[inline(always)]
    pub fn bisect(self) -> (Self, Self) {
        let m = self.midpoint();
        (
            Self::from_bounds_unchecked(self.lo, m),
            Self::from_bounds_unchecked(m, self.hi),
        )
    }
}

impl<V: IntervalFloatVector, W: WideningPolicy> thermite::const_default::ConstDefault for Interval<V, W> {
    const DEFAULT: Self = Interval {
        lo: V::ZERO,
        hi: V::ZERO,
        _widen: PhantomData,
    };
}

// An interval is a wide composite: a constant's interval is degenerate
// (hi == lo, so the augmentation carries no information), which means constants
// and coefficient tables live in the inner vector's primal, recursively.
impl<V: IntervalFloatVector + PrimalProjection, W: WideningPolicy> PrimalProjection for Interval<V, W> {
    type Primal = V::Primal;

    #[inline(always)]
    fn from_primal(p: Self::Primal) -> Self {
        Self::degenerate(V::from_primal(p))
    }

    /// The midpoint, matching [`Interval::midpoint`]: lossy by design, like
    /// dropping a `Dual`'s derivatives.
    #[inline(always)]
    fn to_primal(self) -> Self::Primal {
        self.midpoint().to_primal()
    }
}
