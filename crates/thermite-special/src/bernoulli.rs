//! The Bernoulli sequence as vectors: `$B_0, B_1, B_2, B_3, \ldots$`, splatted.
//!
//! This is the public face of the Bernoulli tables. The raw per-format tables live in
//! [`tables::bernoulli`](crate::tables::bernoulli) and hold only the even-index numbers
//! from `$B_2$` up, because those are the ones that need a table; [`BernoulliSequence`]
//! puts the head and the zeros back so the whole sequence can be iterated.
//!
//! ```
//! use thermite::prelude::*;
//! use thermite_special::bernoulli::BernoulliMath;
//!
//! type V = Vector<f64>;
//!
//! let b: Vec<f64> = V::bernoulli_numbers(-0.5)   // B_1 = -1/2, the caller's choice
//!     .take(7)
//!     .map(|v| v.extract::<0>())
//!     .collect();
//!
//! assert_eq!(b, [1.0, -0.5, 1.0 / 6.0, 0.0, -1.0 / 30.0, 0.0, 1.0 / 42.0]);
//! ```
//!
//! # `$B_1$` is yours to pick
//!
//! It is the one Bernoulli number the two conventions disagree on (`$-1/2$` from the
//! generating function `$x/(e^x - 1)$`, `$+1/2$` from `$x/(1 - e^{-x})$`), so the table
//! does not carry it and this iterator takes it as an argument instead. Pass whichever
//! your formula assumes. Nothing else in the sequence changes with the choice.
//!
//! # Where it stops
//!
//! After the last `$B_{2n}$` representable in the element type: `$B_{258}$` for `f64`
//! (259 items) and `$B_{64}$` for `f32` (65 items). `$|B_{2n}|$` grows factorially, so
//! there is nothing beyond it to yield. See the
//! [table docs](crate::tables::bernoulli) for the boundary in full.
//!
//! The iterator is [`ExactSizeIterator`], so `len()` gives that count up front.

use core::iter::FusedIterator;

use crate::RealPrimalMath;

pub use crate::tables::bernoulli::{BernoulliNumbers, bernoulli_b2n};

/// The Bernoulli sequence `$B_0, B_1, B_2, \ldots$` as splatted vectors, including the
/// zero-valued odd terms.
///
/// Built by [`BernoulliMath::bernoulli_numbers`] or [`BernoulliSequence::new`]. See the
/// [module docs](self) for the `$B_1$` convention and where the sequence ends.
#[derive(Debug, Clone, Copy)]
pub struct BernoulliSequence<V: RealPrimalMath<Element: BernoulliNumbers>> {
    /// The subscript of the next number to yield, so `idx` IS `n` in `$B_n$`, not an
    /// index into the underlying table, which skips `$B_0$`, `$B_1$` and the odd zeros.
    idx: usize,
    /// The caller's `$B_1$`. Held for the whole iteration rather than consumed at step
    /// two, which is what lets the state be a bare counter.
    b1: V::Element,
}

impl<V: RealPrimalMath<Element: BernoulliNumbers>> BernoulliSequence<V> {
    /// Starts at `$B_0$`, with `b1` as the value of `$B_1$`.
    #[inline]
    #[must_use]
    pub const fn new(b1: V::Element) -> Self {
        Self { idx: 0, b1 }
    }
}

impl<V: RealPrimalMath<Element: BernoulliNumbers>> Iterator for BernoulliSequence<V> {
    type Item = V;

    #[inline]
    fn next(&mut self) -> Option<V> {
        let table = <V::Element as BernoulliNumbers>::B2N;

        let n = self.idx;
        if n > 2 * table.len() {
            // Deliberately does NOT advance, so the iterator is fused and `idx` cannot
            // run away past the end.
            return None;
        }
        self.idx = n + 1;

        Some(match n {
            0 => V::ONE,
            1 => V::splat(self.b1),
            // Every odd Bernoulli number past B_1 is zero. The table skips them, which is
            // why `idx` is the subscript and the table index is derived, never the reverse.
            _ if n % 2 == 1 => V::ZERO,
            _ => V::splat(table[n / 2 - 1]),
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let end = 2 * <V::Element as BernoulliNumbers>::B2N.len() + 1;
        let remaining = end.saturating_sub(self.idx);
        (remaining, Some(remaining))
    }
}

impl<V: RealPrimalMath<Element: BernoulliNumbers>> ExactSizeIterator for BernoulliSequence<V> {}

impl<V: RealPrimalMath<Element: BernoulliNumbers>> FusedIterator for BernoulliSequence<V> {}

/// Bernoulli numbers for real primal vectors.
///
/// Blanket-implemented for every [`RealPrimalMath`] vector whose element carries the
/// tables, so bringing the trait into scope is all that is needed.
pub trait BernoulliMath: RealPrimalMath<Element: BernoulliNumbers> + Sized {
    /// The sequence `$B_0, B_1, B_2, \ldots$` as splatted vectors, zeros included.
    ///
    /// `b1` is the value of `$B_1$`, which the tables deliberately do not choose for you.
    /// See the [module docs](self).
    #[inline]
    #[must_use]
    fn bernoulli_numbers(b1: Self::Element) -> BernoulliSequence<Self> {
        BernoulliSequence::new(b1)
    }
}

impl<V: RealPrimalMath<Element: BernoulliNumbers>> BernoulliMath for V {}
