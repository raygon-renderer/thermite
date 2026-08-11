//! The gamma family's backend for [`Compensated`], one rung below
//! [`SpecializedSpecialMath`](thermite_special::specialized::SpecializedSpecialMath).
//!
//! See [the parent module](super) for why this rung exists. In short: the gamma family
//! is the part of `thermite-special` that is driven by fitted coefficients, and the
//! width of a `Compensated` decides which coefficients are correct - so it needs a
//! dispatch axis that `Compensated`'s single blanket backend impl does not have.
//!
//! # Implementing
//!
//! Every method has a default, so the minimal impl is empty:
//!
//! ```ignore
//! impl<V: FloatVector<Element = f64>> SpecializedCompensatedSpecialMath<Compensated<f64>>
//!     for Compensated<V> {}
//! ```
//!
//! Override a method when this width can do better than the generic series - typically
//! by carrying a table tuned to it.
//!
//! # The generic algorithm
//!
//! The defaults are Stirling and its derivatives, which is one expansion in one set of
//! constants for the whole family. Shift the argument up by the recurrences until it is
//! large (`$x \gtrsim 30$` for double-double), then:
//!
//! ```math
//! \ln\Gamma(x) \sim (x - \tfrac{1}{2})\ln x - x + \tfrac{1}{2}\ln 2\pi
//!     + \sum_{n \ge 1} \frac{B_{2n}}{2n(2n-1)x^{2n-1}}
//! ```
//! ```math
//! \psi(x) \sim \ln x - \frac{1}{2x} - \sum_{n \ge 1} \frac{B_{2n}}{2n\,x^{2n}}
//! \qquad
//! \psi_1(x) \sim \frac{1}{x} + \frac{1}{2x^2} + \sum_{n \ge 1} \frac{B_{2n}}{x^{2n+1}}
//! ```
//!
//! The Bernoulli numbers are exact rationals, so unlike the minimax rationals the real
//! `f32`/`f64` paths use, they extend to any precision without refitting - there is no
//! oracle to chase and no table to source. Roughly 13 terms clear `$2^{-106}$` at
//! `$x > 30$`, about half that for double-single, so the term count is a `const` off the
//! mantissa width rather than a fixed loop.
//!
//! This is why `Compensated` may never want Lanczos. The real paths use it because it
//! skips the shift loop; here every operation is already an order of magnitude more
//! expensive, so the loop costs relatively less and a 24-coefficient table at 32 digits
//! costs a lot to source and validate.

mod pd;
mod ps;

use thermite::math::policy::Policy;
use thermite::math::{FloatConsts, TranscendentalMathWithPolicy};
use thermite::prelude::*;

use crate::Compensated;

/// What a default body needs of `Compensated<V>` in order to do compensated arithmetic.
///
/// Requested per method rather than as a supertrait of
/// [`SpecializedCompensatedSpecialMath`] - see that trait's docs for why the difference
/// matters.
pub trait CompensatedGammaOps: FloatVector + TranscendentalMathWithPolicy {}
impl<T> CompensatedGammaOps for T where T: FloatVector + TranscendentalMathWithPolicy {}

/// Argument the shift loop drives `z` up to before the asymptotic series is used.
///
/// 30 rather than 20 trades ten more shift steps for two fewer series terms: the
/// double-double case needs 13 coefficients at 20 and 11 at 30. Shift steps are one
/// multiply or divide each, series terms are a multiply-add plus a constant, and going
/// further out (40) starts costing more in the loop than it saves in the tail.
///
/// Numerator size is *not* a constraint on this choice - `CompensatedConstRatio`
/// evaluates the ratio in f64 before splitting it across the two limbs, so a coefficient
/// like B_24's 236364091 is carried exactly even at double-single width.
const SHIFT_TARGET: i64 = 30;

/// Compile-time rational `N/D`, split across both limbs.
///
/// `CompensatedConstRatio` does the splitting at const time, so the coefficients cost no
/// runtime division and carry the full double-double value of the ratio - not the ratio
/// rounded to the element type first.
#[inline(always)]
fn frac<C: FloatVector, const N: i64, const D: i64>() -> C {
    C::splat(const { <C::Element as FloatElement>::ConstRatio::<N, D>::VALUE })
}

/// Compile-time integer `N`, same mechanism.
#[inline(always)]
fn int_frac<C: FloatVector, const N: i64>() -> C {
    C::splat(const { <C::Element as FloatElement>::ConstInt::<N>::VALUE })
}

/// `$\sum_{n\ge1} rac{B_{2n}}{2n(2n-1)} w^{n-1}$`, Horner in `$w = 1/z^2$`.
///
/// The coefficients are exact rationals of small integers - no floating-point literals
/// anywhere - so one table serves every width. Emitted highest-order first, which is
/// what Horner wants and also what makes truncating cheap: a narrower type only needs
/// the last few, and dropping leading terms is exactly what starting the accumulator at
/// zero does.
///
/// Double-double needs all eleven at `z >= 30`; double-single needs three. Both are
/// evaluated for now, which costs the narrow case a few multiply-adds it does not need.
#[inline(always)]
fn stirling_series<C: FloatVector>(w: C) -> C {
    let mut acc = <C as NumericVector>::ZERO;

    macro_rules! horner {
        ($(($n:literal, $d:literal)),* $(,)?) => {
            $( acc = acc.mul_add(w, frac::<C, $n, $d>()); )*
        };
    }

    horner!(
        (77683, 5796),
        (-174611, 125400),
        (43867, 244188),
        (-3617, 122400),
        (1, 156),
        (-691, 360360),
        (1, 1188),
        (-1, 1680),
        (1, 1260),
        (-1, 360),
        (1, 12),
    );

    acc
}

/// `$\sum_{n\ge1} rac{B_{2n}}{2n} w^{n-1}$` for digamma, Horner in `$w = 1/z^2$`.
///
/// Eleven terms, the same reach as the Stirling series at the same shift target.
#[inline(always)]
fn digamma_series<C: FloatVector>(w: C) -> C {
    let mut acc = <C as NumericVector>::ZERO;

    macro_rules! horner {
        ($(($n:literal, $d:literal)),* $(,)?) => {
            $( acc = acc.mul_add(w, frac::<C, $n, $d>()); )*
        };
    }

    horner!(
        (77683, 276),
        (-174611, 6600),
        (43867, 14364),
        (-3617, 8160),
        (1, 12),
        (-691, 32760),
        (1, 132),
        (-1, 240),
        (1, 252),
        (-1, 120),
        (1, 12),
    );

    acc
}

/// `$\sum_{n\ge1} B_{2n} w^{n-1}$` for trigamma, Horner in `$w = 1/z^2$`.
///
/// Twelve terms rather than eleven: trigamma's coefficients are the bare Bernoulli
/// numbers, without the `$1/2n$` or `$1/2n(2n-1)$` damping the other two series get, so
/// the tail decays one term slower.
#[inline(always)]
fn trigamma_series<C: FloatVector>(w: C) -> C {
    let mut acc = <C as NumericVector>::ZERO;

    macro_rules! horner {
        ($(($n:literal, $d:literal)),* $(,)?) => {
            $( acc = acc.mul_add(w, frac::<C, $n, $d>()); )*
        };
    }

    horner!(
        (-236364091, 2730),
        (854513, 138),
        (-174611, 330),
        (43867, 798),
        (-3617, 510),
        (7, 6),
        (-691, 2730),
        (5, 66),
        (-1, 30),
        (1, 42),
        (-1, 30),
        (1, 6),
    );

    acc
}

/// Backend for the coefficient-bearing part of `Compensated`'s special math.
///
/// # Implemented on the inner vector, not on `Compensated`
///
/// This is implemented for `V`, with `E = V::Element` (`f32` or `f64`) as the dispatch
/// tag, and its methods take `Compensated<Self>` by argument rather than by `self`. That
/// looks backwards for a math trait and is load-bearing.
///
/// The natural spelling - implement it for `Compensated<V>`, take `self`, and give it
/// `FloatVector<Element = E>` as a supertrait so that default bodies can do arithmetic -
/// does not work. Naming that supertrait asserts the projection
/// `<Compensated<V> as GenericVector>::Element == Compensated<V::Element>` at every use
/// of the bound, and `crate::special`'s seam then normalizes through *that* rather than
/// through `CompensatedFloatVector`, losing the `Mask: CastMask<..>` obligations its
/// `erf` / `erfinv` / `lambert_w` bodies depend on. It surfaces a hundred lines away as
/// unrelated `mismatched types` errors in code that was never touched.
///
/// Hanging the trait off `V` avoids that entirely: the seam then constrains `V`, which
/// cannot say anything about `Compensated<V>`'s projections. What a default body needs
/// is requested per method via [`CompensatedGammaOps`], scoped to that method alone -
/// which is what makes real default bodies possible here at all.
///
/// The element parameter is also what keeps the two per-width impls from colliding:
/// without it both would be `impl<V> .. for V`, differing only in `V::Element`, which
/// coherence does not accept as disjoint.
pub trait SpecializedCompensatedSpecialMath<E>: Sized {
    /// `$\Gamma(x)$`.
    ///
    /// Exponentiates [`compensated_lgamma_r`](Self::compensated_lgamma_r) rather than
    /// running its own reduction, which costs a few bits and saves a second copy of the
    /// reflection: an absolute error `d` in `$\ln\Gamma$` is a *relative* error `d` in
    /// `$\Gamma$`, so the loss is `$\log_2|\ln\Gamma(x)|$` bits - about 6 near x = 30 and
    /// 10 at the overflow edge, out of 106. Avoiding it entirely means a direct Stirling
    /// for `$\Gamma$`, which is only worth writing if those bits are ever missed.
    #[inline(always)]
    fn compensated_tgamma<P: Policy>(x: Compensated<Self>) -> Compensated<Self>
    where
        Compensated<Self>: CompensatedGammaOps,
    {
        let (lg, sign) = Self::compensated_lgamma_r::<P>(x);

        sign * lg.exp_p::<P>()
    }

    /// `$(\ln|\Gamma(x)|, \operatorname{sign}\Gamma(x))$`.
    ///
    /// The sign is carried separately because `lgamma` discards it and `beta` needs it.
    ///
    /// Shift-and-Stirling, with the reflection below `1/2`. See the module docs for the
    /// expansion and for why the shift target is 30.
    #[inline(always)]
    fn compensated_lgamma_r<P: Policy>(x: Compensated<Self>) -> (Compensated<Self>, Compensated<Self>)
    where
        Compensated<Self>: CompensatedGammaOps,
    {
        let one = <Compensated<Self> as NumericVector>::ONE;
        let half = frac::<Compensated<Self>, 1, 2>();

        // Below 1/2 the series is useless, so evaluate at 1 - x and reflect afterwards.
        let reflect = x.cmp_lt(half);
        let z0 = reflect.select(one - x, x);

        // Shift up to the target, accumulating the divided-out product rather than its
        // log: one `ln` at the end instead of thirty. z0 >= 1/2 here, so 30 steps always
        // suffice, and the product tops out around 3e31 - nowhere near overflow.
        let target = int_frac::<Compensated<Self>, SHIFT_TARGET>();
        let mut z = z0;
        let mut prod = one;

        let mut i = 0;
        while i < SHIFT_TARGET {
            let shifting = z.cmp_lt(target);
            prod = prod.mul_c(shifting, z);
            z = z.add_c(shifting, one);
            i += 1;
        }

        // Stirling: (z - 1/2) ln z - z + ln(2pi)/2 + poly(1/z^2)/z
        let w = one / (z * z);
        let poly = stirling_series::<Compensated<Self>>(w);

        let half_ln_tau = (<Compensated<Self> as FloatConsts>::LN_2 + <Compensated<Self> as FloatConsts>::LN_PI) * half;
        let stirling = (z - half).mul_add(z.ln_p::<P>(), half_ln_tau - z) + poly / z;

        let lg = stirling - prod.ln_p::<P>();

        // Reflection: ln|Gamma(x)| = ln(pi) - ln|sin(pi x)| - ln|Gamma(1 - x)|, and
        // sign(Gamma(x)) = sign(sin(pi x)) since Gamma(1 - x) > 0 for x < 1/2. The poles
        // at the non-positive integers fall out on their own: sin(pi x) is zero there, so
        // the log is -inf and the result is +inf.
        let sp = x.sin_pi_p::<P>();
        let reflected = (<Compensated<Self> as FloatConsts>::LN_PI - sp.abs().ln_p::<P>()) - lg;

        let mut value = reflect.select(reflected, lg);
        let sign = one.neg_c(reflect & sp.is_negative());

        // The poles at the non-positive integers have to be selected in rather than left
        // to `ln(0) = -inf` propagating through the reflection. Infinities do not survive
        // compensated arithmetic: `two_sum(finite, inf)` evaluates `inf - inf` while
        // forming the error word, so the pair normalizes to NaN rather than to infinity.
        let zero = <Compensated<Self> as NumericVector>::ZERO;
        let is_pole = reflect & x.cmp_le(zero) & x.cmp_eq(x.floor());
        value = is_pole.select(<Compensated<Self> as FloatVector>::INFINITY, value);

        (value, sign)
    }

    /// `$\psi(x)$`, the digamma function.
    ///
    /// Same shape as [`compensated_lgamma_r`](Self::compensated_lgamma_r) - shift up,
    /// then the asymptotic series - but the recurrence `$\psi(x) = \psi(x+1) - 1/x$`
    /// accumulates a *sum* of reciprocals rather than a product, so it cannot be deferred
    /// to a single log at the end.
    #[inline(always)]
    fn compensated_digamma<P: Policy>(x: Compensated<Self>) -> Compensated<Self>
    where
        Compensated<Self>: CompensatedGammaOps,
    {
        let one = <Compensated<Self> as NumericVector>::ONE;
        let half = frac::<Compensated<Self>, 1, 2>();

        let reflect = x.cmp_lt(half);
        let z0 = reflect.select(one - x, x);

        let target = int_frac::<Compensated<Self>, SHIFT_TARGET>();
        let mut z = z0;
        let mut acc = <Compensated<Self> as NumericVector>::ZERO;

        let mut i = 0;
        while i < SHIFT_TARGET {
            let shifting = z.cmp_lt(target);
            acc = acc.add_c(shifting, one / z);
            z = z.add_c(shifting, one);
            i += 1;
        }

        // psi(z) ~ ln z - 1/(2z) - sum B_2n/(2n z^2n), the sum being w * horner(w).
        let w = one / (z * z);
        let psi = (z.ln_p::<P>() - half / z) - w * digamma_series::<Compensated<Self>>(w);

        let value = psi - acc;

        // psi(x) = psi(1 - x) - pi cot(pi x). Both halves of the cotangent come out of one
        // reduction, and it is exactly the poles of `sin_pi` that carry psi's own poles.
        let (sp, cp) = x.sincos_pi_p::<P>();
        let reflected = value - <Compensated<Self> as FloatConsts>::PI * (cp / sp);

        reflect.select(reflected, value)
    }

    /// `$\psi_1(x)$`, the trigamma function.
    ///
    /// As [`compensated_digamma`](Self::compensated_digamma), with the recurrence
    /// `$\psi_1(x) = \psi_1(x+1) + 1/x^2$` and the reflection
    /// `$\psi_1(x) + \psi_1(1-x) = \pi^2/\sin^2(\pi x)$`. Note the reflection *adds*
    /// rather than subtracting, unlike digamma's.
    #[inline(always)]
    fn compensated_trigamma<P: Policy>(x: Compensated<Self>) -> Compensated<Self>
    where
        Compensated<Self>: CompensatedGammaOps,
    {
        let one = <Compensated<Self> as NumericVector>::ONE;
        let half = frac::<Compensated<Self>, 1, 2>();

        let reflect = x.cmp_lt(half);
        let z0 = reflect.select(one - x, x);

        let target = int_frac::<Compensated<Self>, SHIFT_TARGET>();
        let mut z = z0;
        let mut acc = <Compensated<Self> as NumericVector>::ZERO;

        let mut i = 0;
        while i < SHIFT_TARGET {
            let shifting = z.cmp_lt(target);
            acc = acc.add_c(shifting, one / (z * z));
            z = z.add_c(shifting, one);
            i += 1;
        }

        // psi_1(z) ~ (1 + 1/(2z) + w*horner(w)) / z, with w = 1/z^2.
        let w = one / (z * z);
        let psi1 = (one + half / z + w * trigamma_series::<Compensated<Self>>(w)) / z;

        let value = psi1 + acc;

        let sp = x.sin_pi_p::<P>();
        let reflected = <Compensated<Self> as FloatConsts>::PI_SQUARED / (sp * sp) - value;

        reflect.select(reflected, value)
    }

    /// `$B(a, b) = \Gamma(a)\Gamma(b)/\Gamma(a+b)$`.
    ///
    /// Through logs rather than as a ratio of gammas, which overflows for arguments the
    /// beta function itself handles perfectly well. Rides entirely on
    /// [`compensated_lgamma_r`](Self::compensated_lgamma_r), so a width that overrides
    /// that one gets this for free and should never need to touch this.
    #[inline(always)]
    fn compensated_beta<P: Policy>(a: Compensated<Self>, b: Compensated<Self>) -> Compensated<Self>
    where
        Compensated<Self>: CompensatedGammaOps,
    {
        let (la, sa) = Self::compensated_lgamma_r::<P>(a);
        let (lb, sb) = Self::compensated_lgamma_r::<P>(b);
        let (lab, sab) = Self::compensated_lgamma_r::<P>(a + b);

        ((la + lb) - lab).exp_p::<P>() * ((sa * sb) / sab)
    }
}
