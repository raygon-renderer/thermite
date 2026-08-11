//! Double-single: `Compensated<V>` over an `f32` element, ~48 bits of mantissa.
//!
//! Named after `thermite-special`'s `ps.rs`, and for the same reason - this is the
//! `f32`-element half of the per-width split.
//!
//! Empty, so every method takes the generic Stirling/Bernoulli default from
//! [`super`]. Two things would justify filling it in, neither of them urgent:
//!
//! - ~48 bits is inside `lanczos13m53`'s range, so the coefficients `thermite-special`
//!   already ships as `LANCZOS_F64` would serve here if re-expressed as
//!   `Compensated<f32>` constants - trading the shift loop for a table it already owns.
//! - The Bernoulli series needs about half as many terms at this width as at
//!   double-double, which the default's term count should derive rather than assume.

use thermite::prelude::*;

use super::SpecializedCompensatedSpecialMath;

impl<V: FloatVector<Element = f32>> SpecializedCompensatedSpecialMath<f32> for V {}
