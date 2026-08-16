//! Scalarized `libm` fallbacks for [`PrecisionPolicy::Reference`].
//!
//! [`PrecisionPolicy::Reference`] is the one tier that is not trying to be fast. Its
//! contract is that the result is bit-identical to the `libm` crate (the Rust port of
//! musl's libm) on every backend, every lane count, and every target, for the subset of
//! functions libm actually provides. That makes it usable two ways: as a differential
//! oracle for the faster tiers, and as an on-ramp for callers who want Thermite's
//! generic plumbing while keeping the exact numerics they already have.
//!
//! The mechanism is a lane loop: extract, call the scalar `libm` entry point, insert.
//! There is no vectorization here by design, so a wide vector at this tier costs `LANES`
//! scalar calls. On the 1-lane scalar backend the loop has a single iteration and folds
//! to a direct `libm` call, which is why instantiating a generic kernel at
//! `Vector<f32>` + `Reference` is no slower than calling `libm` by hand.
//!
//! Functions with no `libm` counterpart (`sinc`, `sin_pi`, `exph`, `ln1m_expnx`, ...)
//! have no reference arm and fall through to the normal implementation, which at this
//! tier is already the most accurate one Thermite has.

use super::{Policy, PrecisionPolicy};
use crate::vector::GenericVector;

/// True when `P` asks for reference precision.
#[inline(always)]
pub const fn is_reference<P: Policy>() -> bool {
    P::POLICY.precision.eq(PrecisionPolicy::Reference)
}

/// Apply a scalar function lane by lane.
///
/// Hand-rolled loop rather than `array::map`/`from_fn`: those fail to inline inside
/// `#[target_feature]` code and would leave a call per lane on top of the libm call.
#[inline(always)]
pub fn map1<V, F>(x: V, f: F) -> V
where
    V: GenericVector,
    F: Fn(V::Element) -> V::Element,
{
    let mut out = x;
    let mut i = 0;
    while i < V::LANES {
        out = out.insertv(i, f(x.extractv(i)));
        i += 1;
    }
    out
}

/// Two-argument form of [`map1`], for `atan2`/`powf`/`hypot`.
#[inline(always)]
pub fn map2<V, F>(x: V, y: V, f: F) -> V
where
    V: GenericVector,
    F: Fn(V::Element, V::Element) -> V::Element,
{
    let mut out = x;
    let mut i = 0;
    while i < V::LANES {
        out = out.insertv(i, f(x.extractv(i), y.extractv(i)));
        i += 1;
    }
    out
}

/// One-in, two-out form of [`map1`], for `sin_cos`/`sinh_cosh`.
#[inline(always)]
pub fn map1x2<V, F>(x: V, f: F) -> (V, V)
where
    V: GenericVector,
    F: Fn(V::Element) -> (V::Element, V::Element),
{
    let (mut a, mut b) = (x, x);
    let mut i = 0;
    while i < V::LANES {
        let (u, v) = f(x.extractv(i));
        a = a.insertv(i, u);
        b = b.insertv(i, v);
        i += 1;
    }
    (a, b)
}
