//! The polylogarithm order.
//!
//! [`polylog`](crate::SpecialMath::polylog) takes its order as a [`PolylogOrder`]: a
//! **scalar**, uniform across the packet, tagged by the class of order it carries.
//! `$\mathrm{Li}_n$` at whole-number `n` is a table lookup: every coefficient of the unity
//! series is a tabulated `$\zeta$` value, the leading `$\Gamma(1-s)(-\mu)^{s-1}$` term
//! collapses into `$H_{n-1} - \ln(-\mu)$`, and the far
//! field is the Bernoulli-polynomial inversion formula. At arbitrary real `s` the same
//! series needs a sweep of live `$\zeta(s-k)$` evaluations, a real power, and Wood's
//! m-th-root identity in the far field. Those are different algorithms with costs an
//! order of magnitude apart.
//!
//! # Why a scalar, not a vector
//!
//! Every order-dependent quantity (`$\zeta(s-k)/k!$`, `$k^{-s}$`, `$\Gamma(1-s)$`, the
//! near-integer brackets) is a _per-call_ scalar precompute, splatted once. A per-lane
//! order would pay that sweep per lane, and the regime choices that depend on `s` (near an
//! integer, non-positive) would become masks over arms every lane then has to evaluate.
//! A caller with several orders runs several calls.
//!
//! # The payload types
//!
//! `PolylogOrder<E, S>` carries the invoking vector's own element types: `E` is its
//! `Element` (`f64` on an `f64` vector, `Complex<f64>` on a complex one, a `Dual` element on
//! a dual one) and `S` is its `Signed` lane element (`i64` on an `f64` vector, `i32` on an
//! `f32` one), so an order is spelled in the arithmetic of the type it is used with and
//! nothing is ever converted. The trait signature is
//! `PolylogOrder<Self::Element, <Self::Signed as GenericVector>::Element>`.
//!
//! # Downgrading
//!
//! [`simplify`](PolylogOrder::simplify) narrows [`Real`](PolylogOrder::Real) to
//! [`Integer`](PolylogOrder::Integer) when the value is _exactly_ whole. Nothing narrower
//! is attempted: unlike the Bessel order there is no half-integer shortcut (Wood's
//! half-integer series is the general one with tabulated constants). A near-integer
//! order is a **correctness** hazard for the general arm rather than a cost choice, which
//! is why the general kernel fuses the two cancelling poles algebraically instead of
//! trusting the caller to have snapped.

use thermite::element::{FloatElement, SignedIntegerElement};
use thermite::math::scalar::Unwrap;
use thermite::prelude::*;

/// The order `$s$` of a polylogarithm, tagged with the class of order it carries. See the
/// [module documentation](self).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PolylogOrder<E, S> {
    /// `$s = n$`, a whole number of either sign, in the vector's signed lane type. The
    /// tabulated arm for `$n \ge 1$`, closed forms at `0` and `1`, the reflected series for
    /// `$n < 0$`.
    Integer(S),

    /// Arbitrary real `$s$`, in the vector's element type. The general algorithm.
    Real(E),
}

impl<E: FloatElement, S: SignedIntegerElement> PolylogOrder<E, S> {
    /// Narrow [`Real`](Self::Real) to [`Integer`](Self::Integer) when the value is exactly
    /// a whole number. Never changes the value of `$s$`.
    ///
    /// `V` is any real vector over `E` (the one the order is about to be used with) and
    /// only supplies the float-to-integer lane conversion.
    #[inline(always)]
    pub fn simplify<V>(self) -> Self
    where
        V: FloatVector<Element = E>,
        V::Signed: GenericVector<Element = S>,
    {
        match self {
            Self::Integer(_) => self,
            Self::Real(s) => {
                let r = FloatElement::round(s);
                if r == s {
                    Self::Integer(V::splat(r).to_signed_integer().extract::<0>())
                } else {
                    self
                }
            }
        }
    }
}

/// The order is a scalar in both the vector and the scalar spelling of `polylog`, so it
/// crosses the scalar surface unchanged.
impl<E, S> Unwrap for PolylogOrder<E, S> {
    type Unwrapped = Self;

    #[inline(always)]
    fn wrap(value: Self) -> Self {
        value
    }

    #[inline(always)]
    fn unwrap(self) -> Self {
        self
    }
}
