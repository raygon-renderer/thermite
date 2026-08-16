//! Double-double: `Compensated<V>` over an `f64` element, ~106 bits of mantissa.
//!
//! Named after `thermite-special`'s `pd.rs`, and for the same reason - this is the
//! `f64`-element half of the per-width split.
//!
//! Empty, so every method takes the generic Stirling/Bernoulli default from [`super`].
//! This is the width where that default is most likely to be the permanent answer: the
//! only table that would beat it is Boost's `lanczos24m113`, whose 24 coefficients have
//! to be sourced and validated at 32 digits, against a shift loop whose cost is already
//! small relative to double-double arithmetic. Fill this in only if a measurement asks
//! for it.

use thermite::prelude::*;

use super::SpecializedCompensatedSpecialMath;
use crate::Compensated;

impl<V: FloatVector<Element = f64>> SpecializedCompensatedSpecialMath<f64> for V {
    const INV_LANGEVIN_STEPS: usize = 1;

    #[inline(always)]
    fn dd_const(hi: f64, lo: f64) -> Compensated<Self> {
        Compensated {
            value: Self::splat(hi),
            error: Self::splat(lo),
        }
    }
}
