//! Outward rounding without touching the hardware rounding mode.
//!
//! Three strategies (measured in `bin/interval_probe`):
//!
//! - `bump`: unconditional `next_down`/`next_up` after a nearest-rounded op.
//!   Valid because every primitive is faithfully rounded. IEEE `next_down`
//!   maps a lower bound that overflowed to `+inf` back to `MAX`, so bump is
//!   sound at overflow with no extra work and gets NO clamp.
//! - `scale`: `x -/+ (2eps|x| + min_positive)`. Cheapest throughput, ~3x
//!   looser, and its finite-guard skips infinite lanes, so it NEEDS the
//!   overflow clamp.
//! - `residual`: an error-free transform proves which lanes rounded in which
//!   direction, and only those step. Tightest, exact ops do not widen at all,
//!   and the residual at infinity is NaN (the mask never fires), so it NEEDS
//!   the overflow clamp too.
//!
//! The clamp: a LOWER bound of `+inf` can only have come from a rounded-up
//! overflow (the true value is > MAX but finite unless the inputs were
//! themselves infinite), so it collapses to `MAX`. Symmetrically, an UPPER
//! bound of `-inf` collapses to `MIN`. Bounds that are _legitimately_
//! infinite (an operand was infinite) are handled by clamping only when the
//! pre-rounding inputs were finite. Callers pass the unrounded sum/product,
//! whose infinity is then genuine.

use thermite::prelude::*;

use crate::IntervalFloatVector;

/// Knuth 2Sum, from thermite-compensated's canonical implementation:
/// `s + r == a + b` exactly, `s = fl(a + b)`. Branch-free.
#[inline(always)]
pub(crate) fn two_sum<V: IntervalFloatVector>(a: V, b: V) -> (V, V) {
    thermite_compensated::ScalarValue::two_sum(a, b)
}

/// 2Prod, from thermite-compensated: `p + r == a * b` exactly. Uses hardware
/// FMA when available and the Veltkamp-split construction otherwise, so the
/// residual is NEVER silently zero (a hand-rolled `mul_adde(b, -p)` would be
/// exactly that trap on non-FMA backends). The mul tier still gates residual
/// on `HAS_NATIVE_FMA` as a _performance_ choice (the split form is ~17 ops),
/// but correctness no longer depends on the gate.
#[inline(always)]
pub(crate) fn two_prod<V: IntervalFloatVector>(a: V, b: V) -> (V, V) {
    thermite_compensated::ScalarValue::two_prod(a, b)
}

/// `(q, r)` with `q = RN(a / b)` and `a == q*b + r`, from thermite-compensated.
///
/// Used for the quotient. Every widening tier assumes the endpoint is within half an ulp,
/// and a bare `/` does not guarantee that under `thermite/algebraic-scalar`: `arcp`
/// rewrites `x / c` for a constant `c` into `x * RN(1/c)`, measured at 1.204 ulp for
/// `c = 49.0`, and enclosure fails. `two_quot(a, b).0` pins the strict division for free.
///
/// The remainder is available for a residual-widened division; nothing uses it yet.
#[inline(always)]
pub(crate) fn two_quot<V: IntervalFloatVector>(a: V, b: V) -> (V, V) {
    thermite_compensated::ScalarValue::two_quot(a, b)
}

/// Exact square `(p, r)` with `p + r == a * a`, from thermite-compensated
/// (FMA fast path, Veltkamp fallback).
#[inline(always)]
pub(crate) fn two_square<V: IntervalFloatVector>(a: V) -> (V, V) {
    thermite_compensated::ScalarValue::square(a)
}

/// `2eps*|x| + min_positive`: at least 1 ulp for normal `x`, and enough to
/// escape zero through the entire denormal range (so a flush-to-zero
/// environment still widens past everything it might flush).
#[inline(always)]
fn scale_amount<V: IntervalFloatVector>(x: V) -> V {
    let eps = <V as FloatVector>::EPSILON;
    x.abs().mul_adde(eps + eps, V::MIN_POSITIVE)
}

/// Overflow clamp for the strategies that need it: a non-finite LOWER bound
/// whose unrounded input was finite collapses to `MAX` (the rounded `+inf`
/// hides a true value in `(MAX, inf)`).
#[inline(always)]
fn clamp_lo_overflow<V: IntervalFloatVector>(widened: V, raw: V) -> V {
    // raw == +inf with finite operands means overflow, while raw == -inf is a
    // legitimate lower bound and must stay. NaN raws pass through (poison).
    let overflowed = raw.cmp_eq(V::INFINITY);
    overflowed.select(V::MAX, widened)
}

/// Symmetric clamp for a non-finite UPPER bound.
#[inline(always)]
fn clamp_hi_overflow<V: IntervalFloatVector>(widened: V, raw: V) -> V {
    let overflowed = raw.cmp_eq(V::NEG_INFINITY);
    overflowed.select(V::MIN, widened)
}

// --- bump: unconditional ulp step -------------------------------------------

#[inline(always)]
pub(crate) fn bump_down<V: IntervalFloatVector>(x: V) -> V {
    x.next_down()
}

#[inline(always)]
pub(crate) fn bump_up<V: IntervalFloatVector>(x: V) -> V {
    x.next_up()
}

// --- scale: multiplicative widening, clamped ---------------------------------

#[inline(always)]
pub(crate) fn scale_down<V: IntervalFloatVector>(x: V) -> V {
    let w = x.is_finite().select(x - scale_amount(x), x);
    clamp_lo_overflow(w, x)
}

#[inline(always)]
pub(crate) fn scale_up<V: IntervalFloatVector>(x: V) -> V {
    let w = x.is_finite().select(x + scale_amount(x), x);
    clamp_hi_overflow(w, x)
}

// --- residual: step only where the EFT proves rounding erred, clamped --------

/// Lower-bound widening of `s` given the exact residual `r` of the op that
/// produced it: the rounded result sits above the exact value iff `r < 0`.
#[inline(always)]
pub(crate) fn residual_down<V: IntervalFloatVector>(s: V, r: V) -> V {
    let w = s.next_down_c(r.cmp_lt(V::ZERO));
    clamp_lo_overflow(w, s)
}

/// Upper-bound widening of `s` given the exact residual `r`.
#[inline(always)]
pub(crate) fn residual_up<V: IntervalFloatVector>(s: V, r: V) -> V {
    let w = s.next_up_c(r.cmp_gt(V::ZERO));
    clamp_hi_overflow(w, s)
}
