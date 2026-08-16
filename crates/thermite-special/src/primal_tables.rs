//! Gamma-family coefficient tables lifted into a *primal* vector type.
//!
//! [`tables`](crate::tables) holds the coefficients as bare elements. This module hands
//! them over already splatted into a vector type, selected by that type rather than by
//! the caller. The point is composites: `Complex<Dual<V, N>>` has
//! `Primal = V`, so it reaches the same real table `Complex<V>` does, instead of needing
//! a hand-written adapter that rebuilds the table as `Dual` constants with every
//! augmented field zero.
//!
//! # Why this is keyed on the primal, and what that excludes
//!
//! The trait says "this type can supply the Lanczos/asymptotic coefficients *at its own
//! precision*". That is a claim about the algorithm, not just about storage. A primal
//! whose precision the f64 coefficients cannot feed - `Compensated`, where a
//! double-double built from a 53-bit literal carries a fake tail - deliberately does
//! **not** implement it, and so cannot reach the shared Lanczos bodies at all. The
//! missing impl is the design: that precision tier needs a different approximation
//! (a reduced Stirling series, Spouge, or Lanczos re-derived at higher precision),
//! not the same one with wider arithmetic.
//!
//! # Element parameter
//!
//! `E` is a trait parameter rather than an associated type for the same reason
//! [`SpecializedCoreMath`](thermite::math::specialized::SpecializedCoreMath) carries
//! one: two blanket impls written `V: FloatVector<Element = f32>` and
//! `V: FloatVector<Element = f64>` are disjoint in fact, but coherence cannot prove it
//! through an associated-type binding. Spelling the element structurally makes them
//! different traits.

use thermite::generic_array::sequence::GenericSequence;
use thermite::generic_array::typenum::{U3, U6, U8, U13};
use thermite::generic_array::{ArrayLength, GenericArray};
use thermite::math::PrimalProjection;
use thermite::prelude::*;

use crate::tables::{DIGAMMA_F32, DIGAMMA_F64, LANCZOS_F32, LANCZOS_F64};

/// The Lanczos parameters splatted into `V`, with a type-level length.
///
/// Mirrors [`Lanczos`](crate::tables::Lanczos) field for field; see its docs for the
/// two coefficient orders and which consumer wants which. The length is a
/// [`typenum`](thermite::generic_array::typenum) length so an implementor can state it
/// as an associated type - a `const N: usize` cannot be returned from a trait method
/// without `generic_const_exprs`.
pub struct LanczosPrimal<V, N: ArrayLength> {
    pub g: V,
    pub p_rev: GenericArray<V, N>,
    pub q_rev: GenericArray<V, N>,
    pub p_expg_scaled: GenericArray<V, N>,
    pub q: GenericArray<V, N>,
}

/// A primal vector type that can supply the Gamma-family coefficients at its precision.
///
/// See the module docs for why a type may legitimately decline to implement this.
pub trait GammaPrimalTables<E>: Sized {
    /// Number of Lanczos coefficients (6 for f32, 13 for f64).
    type NLanczos: ArrayLength;

    /// Number of terms in digamma's asymptotic series (3 for f32, 8 for f64).
    type NDigammaLarge: ArrayLength;

    /// The Lanczos parameters, splatted.
    fn lanczos_primal() -> LanczosPrimal<Self, Self::NLanczos>;

    /// `Digamma::p_large`, the asymptotic series valid off the real axis, splatted.
    fn digamma_p_large() -> GenericArray<Self, Self::NDigammaLarge>;

    /// `Re z` past which digamma's asymptotic series is used; the recurrence walks up
    /// to this first.
    fn digamma_shift() -> Self;
}

/// Builds both blanket impls from one table pair. The two differ only in element type,
/// array lengths, and which constants they read.
macro_rules! impl_gamma_primal_tables {
    ($elem:ty, $lanczos:ident, $digamma:ident, $n_lanczos:ty, $n_digamma:ty) => {
        impl<V> GammaPrimalTables<$elem> for V
        where
            V: FloatVector<Element = $elem> + PrimalProjection<Primal = V>,
        {
            type NLanczos = $n_lanczos;
            type NDigammaLarge = $n_digamma;

            #[inline(always)]
            fn lanczos_primal() -> LanczosPrimal<Self, Self::NLanczos> {
                let l = &$lanczos;

                LanczosPrimal {
                    g: V::splat(l.g),
                    p_rev: GenericArray::generate(|i| V::splat(l.p_rev[i])),
                    q_rev: GenericArray::generate(|i| V::splat(l.q_rev[i])),
                    p_expg_scaled: GenericArray::generate(|i| V::splat(l.p_expg_scaled[i])),
                    q: GenericArray::generate(|i| V::splat(l.q[i])),
                }
            }

            #[inline(always)]
            fn digamma_p_large() -> GenericArray<Self, Self::NDigammaLarge> {
                GenericArray::generate(|i| V::splat($digamma.p_large[i]))
            }

            #[inline(always)]
            fn digamma_shift() -> Self {
                V::splat(10.0)
            }
        }
    };
}

impl_gamma_primal_tables!(f32, LANCZOS_F32, DIGAMMA_F32, U6, U3);
impl_gamma_primal_tables!(f64, LANCZOS_F64, DIGAMMA_F64, U13, U8);
