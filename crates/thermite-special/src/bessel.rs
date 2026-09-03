//! The runtime Bessel order, and the cost class it selects.
//!
//! The const-order Bessel entry point ([`bessel_n`](crate::SpecialMath::bessel_n)) takes a
//! whole-number order known at compile time. The runtime form
//! ([`bessel`](crate::SpecialMath::bessel)) takes [`BesselOrder`] instead of a bare order
//! value, because "what order is this?" and "what does that order cost?" are different
//! questions and only the caller can answer the second one cheaply.
//!
//! The family markers ([`J`], [`Y`], [`I`], [`K`], [`Scaled`], and the Airy selectors) live
//! in this module too. See [`BesselFamily`].
//!
//! # Why a tagged order rather than a plain `$\nu$`
//!
//! `$J_\nu$` at whole-number `$\nu$` is a table lookup plus a recurrence. At half-integer
//! `$\nu$` it is elementary: sines and cosines. At arbitrary real `$\nu$` it is Steed's
//! method: two continued fractions and a Temme series, one of which needs `$O(x)$`
//! iterations. Those are three genuinely different algorithms with costs an order of
//! magnitude apart, and **under SIMD the whole packet pays for whichever is selected**. A
//! per-lane choice would make every lane pay every arm.
//!
//! So the class is carried as a _scalar_ tag and the order values as a vector. One packet
//! runs one algorithm, and the caller can see in the type which one they asked for.
//!
//! # Exact by construction
//!
//! Each variant stores a **numerator**, not a rounded `$\nu$`: `HalfInteger(k)` means
//! `$\nu = k/2$` and `Thirds(k)` means `$\nu = k/3$`. This is not decoration. `1/3` is not
//! representable in binary, so a design that stored `$\nu$` as a float and tagged it
//! separately could not distinguish `Thirds(1)` from a nearby real order, and the tag would
//! be a promise the caller could break. Here the tag cannot disagree with the payload.
//!
//! # Downgrading
//!
//! [`simplify`](BesselOrder::simplify) narrows a value to the cheapest variant its data
//! actually needs, so a caller who reaches for [`Real`](BesselOrder::Real) and happens to
//! pass whole numbers gets the fast path anyway. The runtime forms call it themselves. It's
//! public so a caller with a hot loop can hoist it out and pay the check once.
//!
//! The checks run in cost order and cost proportionally to how general the claim was, so
//! [`Integer`](BesselOrder::Integer) checks nothing at all. Downgrading requires the
//! condition to hold in **every** lane. One odd lane keeps the whole packet on the general
//! path. A caller with a genuinely mixed packet can recover the fast path with
//! [`group_by_value`](thermite::vector::PartialOrdVector::group_by_value), which turns a
//! divergent packet into uniform sub-packets.
//!
//! ## `Real` does not downgrade to `Thirds`, deliberately
//!
//! `Real -> Integer` and `Real -> HalfInteger` are exact: whole numbers and halves are both
//! representable, so the downgrade cannot change which function is evaluated.
//!
//! `Real -> Thirds` would not be. `fl(1.0/3.0)` is not `$1/3$`, so snapping it to
//! `Thirds(1)` would silently evaluate a _different_ function than the caller asked for.
//! Close, but wrong, and wrong in a way no test of `Thirds` itself would catch. A caller who
//! wants an exact third writes `Thirds(1)`, which is the whole reason the variant carries a
//! numerator.

use thermite::math::policy::Policy;
use thermite::math::scalar::Unwrap;
use thermite::prelude::*;

use crate::specialized::{SpecializedRealSpecialMath, SpecializedSpecialMath};

// ---- Marker-selected entry points ----------------------------------------------------------
//
// `x.bessel_n::<J, 2>()`, `x.bessel::<Scaled<I>>(BesselOrder::Real(nu))`,
// `x.sph_bessel::<K>(n)`, `x.airy::<Scaled<Ai>>()`. The family is a type parameter
// (it never carries a value) and the order follows the crate's `_n` const / plain runtime
// convention. Each marker's trait impl is the dispatch: there is no family enum and nothing
// evaluates more than it was asked for. The markers reach the per-family hooks on
// `SpecializedSpecialMath`, so a composite that overrides those hooks (`Dual`, `Complex`)
// is reached through them with no marker-level work.

mod sealed {
    pub trait Sealed {}
}

/// Bessel function of the first kind, `$J_\nu$`. The oscillating minimal solution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct J;

/// Bessel function of the second kind, `$Y_\nu$` (Neumann). The oscillating dominant solution.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Y;

/// Modified Bessel function of the first kind, `$I_\nu$`. Grows like `$e^x$`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct I;

/// Modified Bessel function of the second kind, `$K_\nu$`. Decays like `$e^{-x}$`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct K;

/// The exponentially scaled form of a family or Airy function.
///
/// `Scaled(I)` is `$e^{-|x|} I_\nu$`, `Scaled(K)` is `$e^{x} K_\nu$`, `Scaled(Ai)` is
/// `$e^{\zeta}\mathrm{Ai}$` on the positive axis. `Scaled(J)` and `Scaled(Y)` are SciPy's
/// `jve`/`yve`, `$e^{-|\mathrm{Im}\,z|} J_\nu(z)$`: the factor is 1 on the real axis, so on a
/// real vector they are `J` and `Y` unchanged and cost nothing extra. On a complex vector they
/// are the scaled values. The scaled forms are never "the unscaled value times an
/// exponential". Where the kernels are natively scaled they skip a transcendental and stay in
/// range where the unscaled value has overflowed or underflowed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Scaled<F>(pub F);

/// `$\mathrm{Ai}(x)$`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Ai;

/// `$\mathrm{Ai}'(x)$`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AiPrime;

/// `$\mathrm{Bi}(x)$`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Bi;

/// `$\mathrm{Bi}'(x)$`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BiPrime;

/// A Bessel family marker: [`J`], [`Y`], [`I`], [`K`], or one of them under [`Scaled`].
///
/// The methods are the dispatch, one per (cylindrical/spherical) x (const/runtime order)
/// cell, each reaching the matching per-family hook on `SpecializedSpecialMath`. They are
/// `#[doc(hidden)]` because nothing outside the marker layer calls them: the public spelling
/// is [`bessel_n`](crate::SpecialMath::bessel_n) / [`bessel`](crate::SpecialMath::bessel) and
/// the `sph_` pair. Cylindrical const orders are `i32` (the families reflect at negative
/// order), spherical ones `usize`.
pub trait BesselFamily: Copy + sealed::Sealed {
    #[doc(hidden)]
    fn cyl_n<P: Policy, E, V, const N: i32>(x: V) -> V
    where
        V: FloatVector<Element = E> + SpecializedSpecialMath<E>;

    #[doc(hidden)]
    fn cyl_v<P: Policy, E, V>(x: V, order: BesselOrder<V, V::Signed>) -> V
    where
        V: FloatVector<Element = E> + SpecializedSpecialMath<E>;

    #[doc(hidden)]
    fn sph_n<P: Policy, E, V, const N: usize>(x: V) -> V
    where
        V: FloatVector<Element = E> + SpecializedSpecialMath<E>;

    #[doc(hidden)]
    fn sph_v<P: Policy, E, V>(x: V, n: u32) -> V
    where
        V: FloatVector<Element = E> + SpecializedSpecialMath<E>;
}

/// Stamps `BesselFamily` for one marker from the four hooks it selects.
macro_rules! impl_family {
    ($($marker:ty => {
        cyl_n: $cyl_n:ident $(::<$($cn:tt),*>)?,
        cyl_v: $cyl_v:ident $(::<$($cv:tt),*>)?,
        sph_n: $sph_n:ident,
        sph_v: $sph_v:ident,
    })*) => {$(
        impl sealed::Sealed for $marker {}

        impl BesselFamily for $marker {
            #[inline(always)]
            fn cyl_n<P: Policy, E, V, const N: i32>(x: V) -> V
            where
                V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
            {
                <V as SpecializedSpecialMath<E>>::$cyl_n::<P, N $($(, $cn)*)?>(x)
            }

            #[inline(always)]
            fn cyl_v<P: Policy, E, V>(x: V, order: BesselOrder<V, V::Signed>) -> V
            where
                V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
            {
                <V as SpecializedSpecialMath<E>>::$cyl_v::<P $($(, $cv)*)?>(x, order)
            }

            #[inline(always)]
            fn sph_n<P: Policy, E, V, const N: usize>(x: V) -> V
            where
                V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
            {
                <V as SpecializedSpecialMath<E>>::$sph_n::<P, N>(x)
            }

            #[inline(always)]
            fn sph_v<P: Policy, E, V>(x: V, n: u32) -> V
            where
                V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
            {
                <V as SpecializedSpecialMath<E>>::$sph_v::<P>(x, n)
            }
        }
    )*};
}

impl_family! {
    J => { cyl_n: bessel_j, cyl_v: bessel_jv, sph_n: sph_bessel_j_n, sph_v: sph_bessel_j, }
    Y => { cyl_n: bessel_y, cyl_v: bessel_yv, sph_n: sph_bessel_y_n, sph_v: sph_bessel_y, }
    I => { cyl_n: bessel_i, cyl_v: bessel_iv::<false>, sph_n: sph_bessel_i_n, sph_v: sph_bessel_i, }
    K => { cyl_n: bessel_k, cyl_v: bessel_kv::<false>, sph_n: sph_bessel_k_n, sph_v: sph_bessel_k, }
    Scaled<I> => { cyl_n: bessel_i_scaled, cyl_v: bessel_iv::<true>, sph_n: sph_bessel_i_scaled_n, sph_v: sph_bessel_i_scaled, }
    Scaled<K> => { cyl_n: bessel_k_scaled, cyl_v: bessel_kv::<true>, sph_n: sph_bessel_k_scaled_n, sph_v: sph_bessel_k_scaled, }
    // The oscillating pair's scaling is a unit factor on the real axis: the const and
    // spherical cells are the unscaled hooks outright, the runtime cell is the hook a
    // complex vector overrides.
    Scaled<J> => { cyl_n: bessel_j, cyl_v: bessel_jv_scaled, sph_n: sph_bessel_j_n, sph_v: sph_bessel_j, }
    Scaled<Y> => { cyl_n: bessel_y, cyl_v: bessel_yv_scaled, sph_n: sph_bessel_y_n, sph_v: sph_bessel_y, }
}

/// A family with a ratio kernel `$F_\nu / F_{\nu-1}$` and its inverse, for
/// [`bessel_ratio`](crate::RealSpecialMath::bessel_ratio) and its three companions. Only
/// [`I`] today (the von Mises-Fisher quantities). A `K` ratio would be the next member.
///
/// Real vectors only, like the entries it serves: the ratio kernels compare and take
/// absolute values along the real line.
pub trait BesselRatioFamily: BesselFamily {
    #[doc(hidden)]
    fn ratio<P: Policy, E, V>(x: V, nu: V) -> V
    where
        V: FloatVector<Element = E> + SpecializedRealSpecialMath<E>;

    #[doc(hidden)]
    fn inv_ratio<P: Policy, E, V>(r: V, nu: V) -> V
    where
        V: FloatVector<Element = E> + SpecializedRealSpecialMath<E>;

    #[doc(hidden)]
    fn ratio_1m<P: Policy, E, V>(x: V, nu: V) -> V
    where
        V: FloatVector<Element = E> + SpecializedRealSpecialMath<E>;

    #[doc(hidden)]
    fn inv_ratio_1m<P: Policy, E, V>(t: V, nu: V) -> V
    where
        V: FloatVector<Element = E> + SpecializedRealSpecialMath<E>;
}

impl BesselRatioFamily for I {
    #[inline(always)]
    fn ratio<P: Policy, E, V>(x: V, nu: V) -> V
    where
        V: FloatVector<Element = E> + SpecializedRealSpecialMath<E>,
    {
        <V as SpecializedRealSpecialMath<E>>::bessel_i_ratio::<P>(x, nu)
    }

    #[inline(always)]
    fn inv_ratio<P: Policy, E, V>(r: V, nu: V) -> V
    where
        V: FloatVector<Element = E> + SpecializedRealSpecialMath<E>,
    {
        <V as SpecializedRealSpecialMath<E>>::inv_bessel_i_ratio::<P>(r, nu)
    }

    #[inline(always)]
    fn ratio_1m<P: Policy, E, V>(x: V, nu: V) -> V
    where
        V: FloatVector<Element = E> + SpecializedRealSpecialMath<E>,
    {
        <V as SpecializedRealSpecialMath<E>>::bessel_i_ratio_1m::<P>(x, nu)
    }

    #[inline(always)]
    fn inv_ratio_1m<P: Policy, E, V>(t: V, nu: V) -> V
    where
        V: FloatVector<Element = E> + SpecializedRealSpecialMath<E>,
    {
        <V as SpecializedRealSpecialMath<E>>::inv_bessel_i_ratio_1m::<P>(t, nu)
    }
}

/// An Airy selector: [`Ai`], [`AiPrime`], [`Bi`], [`BiPrime`], or one of them under
/// [`Scaled`]. Each asks the kernel for exactly its own output, so the cost is one Bessel
/// pass, the same as the long-form single entries.
pub trait AiryFn: Copy + sealed::Sealed {
    #[doc(hidden)]
    fn eval<P: Policy, E, V, const SCALED: bool>(x: V) -> V
    where
        V: FloatVector<Element = E> + SpecializedSpecialMath<E>;
}

macro_rules! impl_airy_fn {
    ($($marker:ident => $plain:ident / $scaled:ident;)*) => {$(
        impl sealed::Sealed for $marker {}
        impl sealed::Sealed for Scaled<$marker> {}

        impl AiryFn for $marker {
            #[inline(always)]
            fn eval<P: Policy, E, V, const SCALED: bool>(x: V) -> V
            where
                V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
            {
                if const { SCALED } {
                    <V as SpecializedSpecialMath<E>>::$scaled::<P>(x)
                } else {
                    <V as SpecializedSpecialMath<E>>::$plain::<P>(x)
                }
            }
        }

        impl AiryFn for Scaled<$marker> {
            #[inline(always)]
            fn eval<P: Policy, E, V, const SCALED: bool>(x: V) -> V
            where
                V: FloatVector<Element = E> + SpecializedSpecialMath<E>,
            {
                <$marker as AiryFn>::eval::<P, E, V, true>(x)
            }
        }
    )*};
}

impl_airy_fn! {
    Ai => airy_ai / airy_ai_scaled;
    AiPrime => airy_ai_prime / airy_ai_prime_scaled;
    Bi => airy_bi / airy_bi_scaled;
    BiPrime => airy_bi_prime / airy_bi_prime_scaled;
}

/// The order `$\nu$` for the runtime-order Bessel functions, tagged with the class of order
/// it carries. See the [module documentation](self) for why the class is part of the value.
///
/// Variants are listed cheapest first. Every one stores its order **per lane**, so a packet
/// may carry a different order in each lane. What it may not carry is a different _class_.
///
/// `V` is the float vector and `S` its signed-integer companion, in practice always
/// `BesselOrder<V, V::Signed>`, which is what every entry point asks for and what inference
/// produces from a plain `BesselOrder::Integer(k)`. They are separate parameters rather than
/// one because the scalar surface unwraps each payload independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BesselOrder<V, S> {
    /// `$\nu = k$`, a whole number. Fitted minimax rationals at the low orders plus a
    /// recurrence. The cheapest class, and the only one reaching a coefficient table.
    Integer(S),

    /// `$\nu = k/2$`. Half-integer orders are _elementary_: `$J_{1/2}(x) =
    /// \sqrt{2/\pi x}\,\sin x$`, `$I_{1/2}(x) = \sqrt{2/\pi x}\,\sinh x$`,
    /// `$K_{1/2}(x) = \sqrt{\pi/2x}\,e^{-x}$`, and the recurrence builds the rest with no
    /// continued fraction, `$\Gamma$`, or series. This is also the spherical Bessel family,
    /// via `$j_n(x) = \sqrt{\pi/2x}\,J_{n+1/2}(x)$`.
    HalfInteger(S),

    /// `$\nu = k/3$`. The Airy orders: `$\mathrm{Ai}$` and `$\mathrm{Bi}$` are Bessel
    /// functions at `$\nu = \pm 1/3$` and their derivatives at `$\nu = \pm 2/3$`.
    ///
    /// **Costs the same as [`Real`](Self::Real)**, and the variant promises no shortcut.
    /// No library has one, because there is none short of a dedicated minimax fit per order.
    /// What it buys is _exactness_: a caller who writes
    /// `Thirds(1)` gets the correctly-rounded `$1/3$` rather than whatever they typed.
    Thirds(S),

    /// Arbitrary real `$\nu$`. The general algorithm, and the expensive one.
    Real(V),
}

/// Lets the generated scalar surface (`scalar_bessel_jv` and friends) carry an order: each
/// payload unwraps on its own, which is the reason `V` and `S` are separate parameters.
impl<V: Unwrap, S: Unwrap> Unwrap for BesselOrder<V, S> {
    type Unwrapped = BesselOrder<V::Unwrapped, S::Unwrapped>;

    #[inline(always)]
    fn wrap(value: Self::Unwrapped) -> Self {
        match value {
            BesselOrder::Integer(k) => Self::Integer(Unwrap::wrap(k)),
            BesselOrder::HalfInteger(k) => Self::HalfInteger(Unwrap::wrap(k)),
            BesselOrder::Thirds(k) => Self::Thirds(Unwrap::wrap(k)),
            BesselOrder::Real(v) => Self::Real(Unwrap::wrap(v)),
        }
    }

    #[inline(always)]
    fn unwrap(self) -> Self::Unwrapped {
        match self {
            Self::Integer(k) => BesselOrder::Integer(k.unwrap()),
            Self::HalfInteger(k) => BesselOrder::HalfInteger(k.unwrap()),
            Self::Thirds(k) => BesselOrder::Thirds(k.unwrap()),
            Self::Real(v) => BesselOrder::Real(v.unwrap()),
        }
    }
}

impl<V: FloatVector> BesselOrder<V, V::Signed> {
    /// The order as a float vector.
    ///
    /// Exact for [`Integer`](Self::Integer), [`HalfInteger`](Self::HalfInteger) and
    /// [`Real`](Self::Real). **Lossy for [`Thirds`](Self::Thirds)**, necessarily, as thirds are
    /// not binary-representable, which is why the variant stores a numerator in the first
    /// place. Kernels that need an exact third must consume the numerator, not this.
    #[inline(always)]
    pub fn to_real(self) -> V {
        match self {
            Self::Integer(k) => V::from_signed_integer(k),
            Self::HalfInteger(k) => V::from_signed_integer(k) * V::HALF,
            Self::Thirds(k) => V::from_signed_integer(k) / thermite::const_splat!(int <V::Element>: 3),
            Self::Real(v) => v,
        }
    }

    /// The whole-number order, if [`simplify`](Self::simplify) reduces this to
    /// [`Integer`](Self::Integer).
    ///
    /// `None` says the order genuinely is not whole, which is the question every entry point
    /// asks last, after it has checked for the cheaper classes it can serve directly.
    #[inline(always)]
    pub fn as_integer(self) -> Option<V::Signed> {
        match self.simplify() {
            Self::Integer(k) => Some(k),
            _ => None,
        }
    }

    /// Narrow to the cheapest variant this data actually needs.
    ///
    /// Requires the condition to hold in every lane. Never widens, never changes the value
    /// of `$\nu$`, and never turns [`Real`](Self::Real) into [`Thirds`](Self::Thirds). See
    /// the [module documentation](self) for why that last one would be unsound.
    #[inline(always)]
    pub fn simplify(self) -> Self {
        match self {
            Self::Integer(_) => self,

            // k/2 is a whole number exactly when k is even. One AND and a compare. The
            // shift must be arithmetic, since `>>` is logical even on signed vectors.
            Self::HalfInteger(k) => match (k & V::Signed::ONE).cmp_eq(V::Signed::ZERO).all() {
                true => Self::Integer(k.srai::<1>()),
                false => self,
            },

            // k/3 is a whole number exactly when 3 divides k. Done in the float domain to
            // avoid an integer division: if 3 divides k then k/3 is exact and `round`
            // recovers it, and if it does not then the recovered value fails the check.
            Self::Thirds(k) => {
                let three = thermite::const_splat!(int <V::Element>: 3);
                let kf = V::from_signed_integer(k);
                let m = (kf / three).round();

                match (m * three).cmp_eq(kf).all() {
                    true => Self::Integer(m.to_signed_integer()),
                    false => self,
                }
            }

            // Whole numbers and halves are both exactly representable, so neither of these
            // changes which function gets evaluated. Thirds are not, and are not attempted.
            Self::Real(v) => {
                let r = v.round();

                if r.cmp_eq(v).all() {
                    return Self::Integer(r.to_signed_integer());
                }

                let two_v = v + v;
                let h = two_v.round();

                match h.cmp_eq(two_v).all() {
                    true => Self::HalfInteger(h.to_signed_integer()),
                    false => self,
                }
            }
        }
    }
}
