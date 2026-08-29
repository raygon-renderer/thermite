//! The floating-point relative error bound `$\gamma_n$`, shared by both layouts.
//!
//! `$\gamma_n = \frac{n u}{1 - n u}$`, with `$u$` the unit roundoff (half of
//! machine epsilon), bounds the accumulated rounding error of `n` chained
//! floating-point operations.
//!
//! A dot product of `n` terms, one row of a matrix-vector product, is accurate
//! to within a factor of `$\gamma_n$`, which is what the `*_with_error`
//! transforms in both layouts return so a renderer can offset ray origins past
//! self-intersection.
//!
//! See Higham, *Accuracy and Stability of Numerical Algorithms*, S3.1.
//!
//! # Why there are two of these
//!
//! The formula is one line, but it cannot be written once. [`gamma`] is generic
//! over the *vector* layer and [`gamma_scalar`] over the *element* layer, and no
//! single bound covers both: the `u16 -> Self` conversion lives on
//! `FloatElement` (`from_int`), while a vector reaches it only through
//! `V::splat`. A local helper trait blanket-implemented for both
//! `V: FloatVector` and `E: FloatElement` overlaps and fails coherence.
//!
//! Collapsing [`gamma`] to `V::splat(gamma_scalar::<V::Element>(n))` is not a
//! way out either. It is bit-identical for a plain `Vector<R>`, but the crate is
//! generic over any `FloatVector`, and a composite element does not round like
//! its own scalar: `Compensated::EPSILON` is a double-double
//! (`crates/thermite-compensated/src/lib.rs:2633`), so the divide has to happen
//! in the composite's own arithmetic to mean anything.

use thermite::{
    element::{Element, FloatElement},
    math::FloatConsts,
    vector::FloatVector,
};

/// `$\gamma_n$` at the vector layer: one bound per lane, in the vector's own
/// arithmetic.
///
/// See the [module docs](self) for the definition and the reference.
#[inline(always)]
pub fn gamma<V: FloatVector>(n: u16) -> V {
    // EPSILON is machine epsilon, and the unit roundoff is half of it.
    let nu = <V as FloatVector>::EPSILON * V::HALF * V::splat(Element::from_u16(n));

    nu / (V::ONE - nu)
}

/// `$\gamma_n$` at the element layer: a single scalar bound, for the AoS layout
/// where one register is one object and the error is not per-lane.
///
/// See the [module docs](self) for the definition and the reference.
#[inline(always)]
pub fn gamma_scalar<E: FloatElement + FloatConsts>(n: u16) -> E {
    let nu = E::EPSILON * E::from_ratio(1, 2) * E::from_int(n.into());

    nu / (E::ONE - nu)
}
