//! Interval arithmetic: the widening-policy-driven core operations, the
//! `core::ops` operators, and the vector op-trait impls (`MulAddExt`,
//! `AddSubExt`, `Square`, and the masked families).
//!
//! Every strategy decision keys off `W::TIER` (a const), so each tier
//! monomorphizes to straight-line code with no runtime dispatch. Thermite's
//! math `Policy` plays no role here. It tunes approximation _algorithms_
//! in the (future) math library, never the interval bookkeeping.
//!
//! Every operation encloses the exact set result. Empty lanes stay empty,
//! and `0 * inf` product corner cases resolve to the correct set limit
//! (zero) rather than poisoning min/max with NaN.

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};

use thermite::prelude::*;
use thermite::tribool::{self, Tribool};
use thermite::vector::ops::{AddSubExt, MulAddExt, Square};

use crate::round::{
    bump_down, bump_up, residual_down, residual_up, scale_down, scale_up, two_prod, two_square, two_sum,
};
use crate::widen::{WideningPolicy, WideningTier};
use crate::{Interval, IntervalFloatVector};

#[inline(always)]
const fn is_fastest<W: WideningPolicy>() -> bool {
    matches!(W::TIER, WideningTier::Fastest)
}

/// Residual adds only at `Tightest`. The two_sum chain costs 2x serial
/// latency (measured), so `Balanced` skips it.
#[inline(always)]
const fn residual_add<W: WideningPolicy>() -> bool {
    matches!(W::TIER, WideningTier::Tightest)
}

/// Residual multiplies at `Balanced` when hardware FMA makes the residual a
/// single instruction (+11% serial for 2x tightness, measured), and at
/// `Tightest` unconditionally, with the Veltkamp-split product standing in on
/// non-FMA hardware (slower, but that tier promised tightness).
#[inline(always)]
const fn residual_mul<W: WideningPolicy>(has_fma: bool) -> bool {
    match W::TIER {
        WideningTier::Fastest => false,
        WideningTier::Balanced => has_fma,
        WideningTier::Tightest => true,
    }
}

/// Whether `mul_add` builds each endpoint from one fused rounding instead of a full
/// interval multiply followed by a full interval add. Needs two things:
///
/// - An inner `mul_add` that is an instruction rather than the software emulation.
///   Only `False` is known at compile time to emulate. Wasm's `Indeterminate` is
///   one relaxed op on a fusing engine and the correctly rounded emulation on the
///   rest (thermite's canary decides once at runtime). Either way it is a single
///   correct rounding, but the eight emulated corners on a non-fusing engine cost
///   more than the composed multiply-then-add. That is accepted, since the
///   tightness is the same and the engines that matter fuse.
/// - A tier that widens by an unconditional ulp step, since one rounding needs
///   exactly one step. `Tightest` widens by an exact residual instead, and promises
///   that an exact operation does not widen at all. Keeping that promise for a
///   fused form takes the fma's exact residual (Boldo-Muller's ErrFma, three
///   error-free transforms), where multiply-then-add already has it from the
///   `two_prod` and `two_sum` it runs anyway. A later refinement.
#[inline(always)]
const fn fuses_fma<V: IntervalFloatVector, W: WideningPolicy>() -> bool {
    !matches!(V::HAS_NATIVE_FMA, tribool::False) && !matches!(W::TIER, WideningTier::Tightest)
}

/// One corner `(x, m)` of the fused FMA against both endpoints of `a`, each a single
/// rounding of the exact `x*m + a`.
///
/// A NaN candidate is the `0 * inf` corner. Its product is the set limit 0 rather
/// than NaN, so the candidate is exactly `a`, which is also what `mul_interval`
/// makes of it through `unpoison`. The same patch covers `inf + -inf`, where the
/// infinite endpoint of `a` is the true bound.
#[inline(always)]
fn fma_corner<V: IntervalFloatVector>(x: V, m: V, alo: V, ahi: V) -> (V, V) {
    let (l, h) = (x.mul_add(m, alo), x.mul_add(m, ahi));

    (l.is_nan().select(alo, l), h.is_nan().select(ahi, h))
}

// --- the widening-policy-driven core ops --------------------------------------

impl<V: IntervalFloatVector, W: WideningPolicy> Interval<V, W> {
    /// Exact: `-[lo, hi] = [-hi, -lo]`, no rounding, no widening.
    #[inline(always)]
    pub fn negate(self) -> Self {
        Self::from_bounds_unchecked(-self.hi, -self.lo)
    }

    /// Interval addition.
    #[inline(always)]
    pub fn add_interval(self, rhs: Self) -> Self {
        // Empty lanes survive the sum only against finite endpoints: [+inf, -inf]
        // plus a half-line or `entire` hits `inf + -inf = NaN`, and a NaN bound is
        // not empty (`lo > hi` is false), so the poison is applied explicitly. It
        // is computed off the critical path and costs the two final blends.
        let poison = self.is_empty() | rhs.is_empty();

        let (lo, hi) = if const { is_fastest::<W>() } {
            (scale_down(self.lo + rhs.lo), scale_up(self.hi + rhs.hi))
        } else if const { residual_add::<W>() } {
            let (sl, rl) = two_sum(self.lo, rhs.lo);
            let (sh, rh) = two_sum(self.hi, rhs.hi);
            (residual_down(sl, rl), residual_up(sh, rh))
        } else {
            (bump_down(self.lo + rhs.lo), bump_up(self.hi + rhs.hi))
        };

        Self::from_bounds_unchecked(poison.select(V::INFINITY, lo), poison.select(V::NEG_INFINITY, hi))
    }

    /// Interval subtraction: `[lo1 - hi2, hi1 - lo2]`.
    #[inline(always)]
    pub fn sub_interval(self, rhs: Self) -> Self {
        self.add_interval(rhs.negate())
    }

    /// Interval multiplication.
    ///
    /// All four endpoint products through vector min/max, so it is branchless
    /// with no sign case tree. `0 * inf` products (a zero endpoint against an
    /// infinite one) are the set limit 0, not NaN, so they are patched before
    /// min/max.
    #[inline(always)]
    pub fn mul_interval(self, rhs: Self) -> Self {
        let poison = self.is_empty() | rhs.is_empty();

        let res = if const { residual_mul::<W>(matches!(V::HAS_NATIVE_FMA, tribool::True)) } {
            // Widen each product in each direction first, then min/max: every
            // widened-down product is <= its exact product, so the min of the
            // four encloses the exact min (and symmetrically for max).
            let (p0, r0) = two_prod(self.lo, rhs.lo);
            let (p1, r1) = two_prod(self.lo, rhs.hi);
            let (p2, r2) = two_prod(self.hi, rhs.lo);
            let (p3, r3) = two_prod(self.hi, rhs.hi);

            let (p0, p1, p2, p3) = (unpoison(p0), unpoison(p1), unpoison(p2), unpoison(p3));

            Self::from_bounds_unchecked(
                residual_down(p0, r0)
                    .min(residual_down(p1, r1))
                    .min(residual_down(p2, r2))
                    .min(residual_down(p3, r3)),
                residual_up(p0, r0)
                    .max(residual_up(p1, r1))
                    .max(residual_up(p2, r2))
                    .max(residual_up(p3, r3)),
            )
        } else {
            let p0 = unpoison(self.lo * rhs.lo);
            let p1 = unpoison(self.lo * rhs.hi);
            let p2 = unpoison(self.hi * rhs.lo);
            let p3 = unpoison(self.hi * rhs.hi);

            let lo = p0.min(p1).min(p2.min(p3));
            let hi = p0.max(p1).max(p2.max(p3));

            if const { is_fastest::<W>() } {
                Self::from_bounds_unchecked(scale_down(lo), scale_up(hi))
            } else {
                Self::from_bounds_unchecked(bump_down(lo), bump_up(hi))
            }
        };

        // A degenerate zero operand makes every product exactly zero, and the
        // widening must not fatten it: bump/scale would turn `[0, 0]` into
        // `[-5e-324, 5e-324]`. (The residual tier already gets this right since
        // an exact product has a zero residual, but the cheaper tiers do not.)
        // Boost.Interval's `? * Z -> Z` case.
        let exact_zero = (self.lo.is_zero() & self.hi.is_zero()) | (rhs.lo.is_zero() & rhs.hi.is_zero());
        let res = Self::from_bounds_unchecked(exact_zero.select(V::ZERO, res.lo), exact_zero.select(V::ZERO, res.hi));

        Self::from_bounds_unchecked(
            poison.select(V::INFINITY, res.lo),
            poison.select(V::NEG_INFINITY, res.hi),
        )
    }

    /// The fused multiply-add `self * m + a`, as ONE interval operation.
    ///
    /// The enclosure of `{x*m + a}` is `(X*M) + A`, and adding a fixed endpoint of
    /// `A` is monotone, so the bounds are the four corner products offset by one
    /// endpoint of `A`: `a.lo` against every corner for the lower bound, `a.hi` for
    /// the upper. Each candidate is then a single fused multiply-add, one rounding
    /// of the exact `x*m + a`, so the result widens once. Multiplying and then
    /// adding rounds twice and widens twice, so across a chain of them (a dot
    /// product, a Newton step) this halves the accumulated width.
    ///
    /// Falls back to multiplying and then adding where the fused form would buy
    /// nothing (see [`fuses_fma`]). That costs the second rounding and the second
    /// widening, but agrees on every identity below.
    #[inline(always)]
    pub fn mul_add_interval(self, m: Self, a: Self) -> Self {
        let (lo, hi) = if const { fuses_fma::<V, W>() } {
            let (c0, d0) = fma_corner(self.lo, m.lo, a.lo, a.hi);
            let (c1, d1) = fma_corner(self.lo, m.hi, a.lo, a.hi);
            let (c2, d2) = fma_corner(self.hi, m.lo, a.lo, a.hi);
            let (c3, d3) = fma_corner(self.hi, m.hi, a.lo, a.hi);

            let lo = c0.min(c1).min(c2.min(c3));
            let hi = d0.max(d1).max(d2.max(d3));

            // One widening for the one rounding. `next_down`/`next_up` are monotone,
            // so stepping the min/max is the same as stepping every candidate first.
            if const { is_fastest::<W>() } {
                (scale_down(lo), scale_up(hi))
            } else {
                (bump_down(lo), bump_up(hi))
            }
        } else {
            let c = self.mul_interval(m).add_interval(a);
            (c.lo, c.hi)
        };

        // `X * Z + A` is exactly `A` (Boost.Interval's `? * Z -> Z`, carried through
        // the addend). A degenerate zero factor makes every product exactly zero, and
        // the widening must not fatten what it adds. The fused corners already return
        // `A`. This exists for the composed path, which would otherwise disagree by an ulp.
        let exact_zero = (self.lo.is_zero() & self.hi.is_zero()) | (m.lo.is_zero() & m.hi.is_zero());
        let (lo, hi) = (exact_zero.select(a.lo, lo), exact_zero.select(a.hi, hi));

        // Empty in, empty out. This outranks the zero identity above.
        let poison = self.is_empty() | m.is_empty() | a.is_empty();

        Self::from_bounds_unchecked(poison.select(V::INFINITY, lo), poison.select(V::NEG_INFINITY, hi))
    }

    /// Interval division.
    ///
    /// Direct endpoint quotients, never numerator times reciprocal interval
    /// (that rounds twice and correlates). Divisor lanes with zero in their
    /// interior give the entire line `[-inf, +inf]`. A zero on a divisor
    /// endpoint keeps the half-line result of extended division. A residual
    /// form via `fma(q, b, -a)` is a later refinement.
    #[inline(always)]
    pub fn div_interval(self, rhs: Self) -> Self {
        let poison = self.is_empty() | rhs.is_empty();
        let zero_div = rhs.lo.cmp_le(V::ZERO) & rhs.hi.cmp_ge(V::ZERO);

        let q0 = unpoison(self.lo / rhs.lo);
        let q1 = unpoison(self.lo / rhs.hi);
        let q2 = unpoison(self.hi / rhs.lo);
        let q3 = unpoison(self.hi / rhs.hi);

        let lo = q0.min(q1).min(q2.min(q3));
        let hi = q0.max(q1).max(q2.max(q3));

        let (lo, hi) = if const { is_fastest::<W>() } {
            (scale_down(lo), scale_up(hi))
        } else {
            (bump_down(lo), bump_up(hi))
        };

        // Default for a zero-containing divisor is the whole line...
        let lo = zero_div.select(V::NEG_INFINITY, lo);
        let hi = zero_div.select(V::INFINITY, hi);

        // ...but a divisor whose zero sits on an ENDPOINT still bounds the
        // quotient on one side (Boost.Interval's `div_positive`/`div_negative`
        // extended division). Only a zero in the divisor's INTERIOR needs the
        // whole line.
        //
        //   y = [0, yu], x > 0  ->  [xl/yu, +inf]      y = [yl, 0], x > 0  ->  [-inf, xl/yl]
        //   y = [0, yu], x < 0  ->  [-inf, xu/yu]      y = [yl, 0], x < 0  ->  [xu/yl, +inf]
        let y_zero_lo = rhs.lo.is_zero() & !rhs.hi.is_zero(); // zero at the low end only
        let y_zero_hi = rhs.hi.is_zero() & !rhs.lo.is_zero(); // zero at the high end only
        let x_pos = self.lo.cmp_gt(V::ZERO);
        let x_neg = self.hi.cmp_lt(V::ZERO);

        #[inline(always)]
        fn widen_dn<V: IntervalFloatVector, W: WideningPolicy>(v: V) -> V {
            if const { is_fastest::<W>() } {
                scale_down(v)
            } else {
                bump_down(v)
            }
        }

        #[inline(always)]
        fn widen_up<V: IntervalFloatVector, W: WideningPolicy>(v: V) -> V {
            if const { is_fastest::<W>() } {
                scale_up(v)
            } else {
                bump_up(v)
            }
        }

        // For y = [0, yu] the finite bound is x?/yu, and for y = [yl, 0] it is x?/yl.
        let lo = (zero_div & y_zero_lo & x_pos).select(widen_dn::<V, W>(self.lo / rhs.hi), lo);
        let hi = (zero_div & y_zero_lo & x_neg).select(widen_up::<V, W>(self.hi / rhs.hi), hi);
        let hi = (zero_div & y_zero_hi & x_pos).select(widen_up::<V, W>(self.lo / rhs.lo), hi);
        let lo = (zero_div & y_zero_hi & x_neg).select(widen_dn::<V, W>(self.hi / rhs.lo), lo);

        Self::from_bounds_unchecked(poison.select(V::INFINITY, lo), poison.select(V::NEG_INFINITY, hi))
    }

    /// The interval square: `[mig^2, mag^2]`.
    ///
    /// NOT `self * self`. The dependency problem would produce a negative
    /// lower bound for any zero-straddling interval, where the true square is
    /// never negative.
    #[inline(always)]
    pub fn square_interval(self) -> Self {
        let poison = self.is_empty();

        let mig = self.mignitude();
        let mag = self.magnitude();

        let res = if const { residual_mul::<W>(matches!(V::HAS_NATIVE_FMA, tribool::True)) } {
            let (pl, rl) = two_square(mig);
            let (ph, rh) = two_square(mag);
            Self::from_bounds_unchecked(
                // mig^2 is never negative, so the widened bound clamps at 0.
                residual_down(pl, rl).max(V::ZERO),
                residual_up(ph, rh),
            )
        } else if const { is_fastest::<W>() } {
            Self::from_bounds_unchecked(scale_down(mig * mig).max(V::ZERO), scale_up(mag * mag))
        } else {
            Self::from_bounds_unchecked(bump_down(mig * mig).max(V::ZERO), bump_up(mag * mag))
        };

        Self::from_bounds_unchecked(
            poison.select(V::INFINITY, res.lo),
            poison.select(V::NEG_INFINITY, res.hi),
        )
    }

    /// Interval square root.
    ///
    /// Domain-intersected: the negative part of the input is discarded
    /// (standard interval convention), and lanes entirely below zero come
    /// out empty. Hardware `sqrt` is correctly rounded, so the residual form
    /// is the exact `s*s - x` via the split square.
    #[inline(always)]
    pub fn sqrt_interval(self) -> Self {
        let all_negative = self.hi.cmp_lt(V::ZERO);
        let poison = self.is_empty() | all_negative;

        let lo_in = self.lo.max(V::ZERO);
        let sl = lo_in.sqrt();
        let sh = self.hi.sqrt();

        let res = if const { residual_mul::<W>(matches!(V::HAS_NATIVE_FMA, tribool::True)) } {
            // s*s = p + e exactly, and p - x is exact near sqrt (Sterbenz), so
            // r = (p - x) + e is the exact s*s - x. NOTE the sign convention
            // flip versus two_sum residuals: here r > 0 means s > sqrt(x)
            // (the rounded root sits ABOVE the exact one), so both widening
            // calls take -r. Getting the lo negation wrong was a real
            // 0.19-ulp containment hole, caught by the generic-kernel test.
            let (pl, el) = two_square(sl);
            let (ph, eh) = two_square(sh);
            let rl = (pl - lo_in) + el;
            let rh = (ph - self.hi) + eh;
            Self::from_bounds_unchecked(residual_down(sl, -rl).max(V::ZERO), residual_up(sh, -rh))
        } else if const { is_fastest::<W>() } {
            Self::from_bounds_unchecked(scale_down(sl).max(V::ZERO), scale_up(sh))
        } else {
            Self::from_bounds_unchecked(bump_down(sl).max(V::ZERO), bump_up(sh))
        };

        Self::from_bounds_unchecked(
            poison.select(V::INFINITY, res.lo),
            poison.select(V::NEG_INFINITY, res.hi),
        )
    }

    /// The reciprocal `1 / self` (via [`div_interval`](Self::div_interval),
    /// so zero-containing lanes give the entire line).
    #[inline(always)]
    pub fn recip_interval(self) -> Self {
        Self::degenerate(V::ONE).div_interval(self)
    }

    /// The interval absolute value: `[mig, mag]`. Exact, no widening.
    #[inline(always)]
    pub fn abs_interval(self) -> Self {
        let poison = self.is_empty();
        Self::from_bounds_unchecked(
            poison.select(V::INFINITY, self.mignitude()),
            poison.select(V::NEG_INFINITY, self.magnitude()),
        )
    }

    /// Endpoint-wise minimum (the set `{min(a, b) : a in self, b in rhs}`).
    /// Exact, since `min` is monotone in both arguments.
    #[inline(always)]
    pub fn min_interval(self, rhs: Self) -> Self {
        Self::from_bounds_unchecked(self.lo.min(rhs.lo), self.hi.min(rhs.hi))
    }

    /// Endpoint-wise maximum. Exact.
    #[inline(always)]
    pub fn max_interval(self, rhs: Self) -> Self {
        Self::from_bounds_unchecked(self.lo.max(rhs.lo), self.hi.max(rhs.hi))
    }
}

/// `0 * inf` (and `inf - inf` style) artifacts inside legitimate endpoint
/// combinations: the set-limit value is 0, so NaN products are patched to
/// zero. Invalid inputs are handled by the callers' empty masks instead.
#[inline(always)]
fn unpoison<V: IntervalFloatVector>(p: V) -> V {
    p.is_nan().select(V::ZERO, p)
}

// --- core::ops operators ------------------------------------------------------

impl<V: IntervalFloatVector, W: WideningPolicy> Neg for Interval<V, W> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        self.negate()
    }
}

impl<V: IntervalFloatVector, W: WideningPolicy> Add for Interval<V, W> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        self.add_interval(rhs)
    }
}

impl<V: IntervalFloatVector, W: WideningPolicy> Sub for Interval<V, W> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        self.sub_interval(rhs)
    }
}

impl<V: IntervalFloatVector, W: WideningPolicy> Mul for Interval<V, W> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        self.mul_interval(rhs)
    }
}

impl<V: IntervalFloatVector, W: WideningPolicy> Div for Interval<V, W> {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        self.div_interval(rhs)
    }
}

/// Containment-valid but wide: `a - trunc(a/b) * b`, composed from the enclosing
/// ops. Exists because `NumOps` requires it. Do not expect tight remainders.
impl<V: IntervalFloatVector, W: WideningPolicy> Rem for Interval<V, W> {
    type Output = Self;
    #[inline(always)]
    fn rem(self, rhs: Self) -> Self {
        let q = self / rhs;
        let qt = Self::from_bounds_unchecked(q.lo.trunc(), q.hi.trunc());
        self - qt * rhs
    }
}

impl<V: IntervalFloatVector, W: WideningPolicy, T> AddAssign<T> for Interval<V, W>
where
    Self: Add<T, Output = Self>,
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: T) {
        *self = *self + rhs;
    }
}
impl<V: IntervalFloatVector, W: WideningPolicy, T> SubAssign<T> for Interval<V, W>
where
    Self: Sub<T, Output = Self>,
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: T) {
        *self = *self - rhs;
    }
}
impl<V: IntervalFloatVector, W: WideningPolicy, T> MulAssign<T> for Interval<V, W>
where
    Self: Mul<T, Output = Self>,
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}
impl<V: IntervalFloatVector, W: WideningPolicy, T> DivAssign<T> for Interval<V, W>
where
    Self: Div<T, Output = Self>,
{
    #[inline(always)]
    fn div_assign(&mut self, rhs: T) {
        *self = *self / rhs;
    }
}
impl<V: IntervalFloatVector, W: WideningPolicy, T> RemAssign<T> for Interval<V, W>
where
    Self: Rem<T, Output = Self>,
{
    #[inline(always)]
    fn rem_assign(&mut self, rhs: T) {
        *self = *self % rhs;
    }
}

// --- MulAddExt: mul-then-add, both enclosing ---------------------------------
//
// An interval "FMA" is NOT endpoint-FMA over an already-rounded product (Pitfall 4
// in the plan): it is the enclosure of `{x*m + a}`, built in `mul_add_interval` from
// the four corners with a real fused multiply-add per candidate where the inner
// vector has one, and from a full interval multiply followed by a full interval add
// where it does not. The `_e` estimating forms are identical, as there is no cheaper
// valid form to estimate with.
//
// `HAS_NATIVE_FMA` says whether a caller restructuring its arithmetic around
// `mul_add` gains anything, so it forwards `V` for the tiers that fuse and is
// `False` at `Tightest`, which multiplies and adds whatever the hardware offers.

#[rustfmt::skip]
impl<V: IntervalFloatVector, W: WideningPolicy> MulAddExt<Self, Self> for Interval<V, W> {
    type Output = Self;

    const HAS_NATIVE_FMA: Tribool = if fuses_fma::<V, W>() { V::HAS_NATIVE_FMA } else { tribool::False };

    // Negating an interval is exact, so the sign variants are the one kernel with
    // `[-hi, -lo]` inputs rather than four more corner expansions.
    #[inline(always)] fn mul_add(self, m: Self, a: Self) -> Self { self.mul_add_interval(m, a) }
    #[inline(always)] fn mul_sub(self, m: Self, a: Self) -> Self { self.mul_add_interval(m, a.negate()) }
    #[inline(always)] fn nmul_add(self, m: Self, a: Self) -> Self { self.negate().mul_add_interval(m, a) }
    #[inline(always)] fn nmul_sub(self, m: Self, a: Self) -> Self { self.negate().mul_add_interval(m, a.negate()) }

    #[inline(always)] fn mul_adde(self, m: Self, a: Self) -> Self { self.mul_add(m, a) }
    #[inline(always)] fn mul_sube(self, m: Self, a: Self) -> Self { self.mul_sub(m, a) }
    #[inline(always)] fn nmul_adde(self, m: Self, a: Self) -> Self { self.nmul_add(m, a) }
    #[inline(always)] fn nmul_sube(self, m: Self, a: Self) -> Self { self.nmul_sub(m, a) }
}

// --- Square: the dependency-correct interval square --------------------------

impl<V: IntervalFloatVector, W: WideningPolicy> Square for Interval<V, W> {
    type Output = Self;

    #[inline(always)]
    fn square(self) -> Self {
        self.square_interval()
    }
}

// --- AddSubExt: alternating add/sub via an even-lane interval negation -------

/// Negates the _even lanes_ as whole intervals: those lanes get `[-hi, -lo]`.
#[inline(always)]
fn neg_even_interval<V: IntervalFloatVector, W: WideningPolicy>(x: Interval<V, W>) -> Interval<V, W> {
    // addsub(0, w) = [-w0, w1, -w2, ...]: even lanes negative -> even mask.
    let even = thermite::vector::ops::AddSubExt::addsub(V::ZERO, V::ONE).cmp_lt(V::ZERO);
    let neg = x.negate();
    Interval::from_bounds_unchecked(even.select(neg.lo, x.lo), even.select(neg.hi, x.hi))
}

impl<V: IntervalFloatVector, W: WideningPolicy> AddSubExt for Interval<V, W> {
    type Output = Self;

    #[inline(always)]
    fn addsub(self, b: Self) -> Self {
        self + neg_even_interval(b)
    }
    #[inline(always)]
    fn fmaddsub(self, b: Self, c: Self) -> Self {
        self * b + neg_even_interval(c)
    }
    #[inline(always)]
    fn fmsubadd(self, b: Self, c: Self) -> Self {
        self * b - neg_even_interval(c)
    }
}
