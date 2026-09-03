#![allow(clippy::needless_arbitrary_self_type)]

//! Special functions for [`Complex`] (`special` feature).
//!
//! Implements `thermite_special`'s [`SpecializedSpecialMath`], giving complex
//! vectors the [`SpecialMath`](thermite_special::SpecialMath) API.
//!
//! `thermite-special` splits its families along the line this crate needs:
//! `Special` is documented as valid for real and complex vectors alike, while
//! `RealSpecial` (`erfinv`, `probit`, `gelu`, `swish`, `algebraic_sigmoid`,
//! `lgamma_r`, ...) and `RealPrimal` (the `_d` forms) are real-only. `Complex`
//! implements the first and not the others, as it implements
//! [`CoreMath`](thermite::math::CoreMath) but not
//! [`RealMath`](thermite::math::RealMath).
//!
//! [`erf`](SpecializedSpecialMath::erf) and [`erfc`](SpecializedSpecialMath::erfc)
//! are implemented over the whole plane. The holomorphic defaults (`hermite`,
//! `hermitev`, `chebyshev`, `jacobi`, `legendre`, `gaussian`) are complex
//! polynomial recurrences and are inherited as they are.
//! [`logistic_sigmoid`](SpecializedSpecialMath::logistic_sigmoid) and
//! [`softplus`](SpecializedSpecialMath::softplus) must be overridden: their
//! defaults are stabilized for the real axis with `|x|` and `max(x, 0)`.
//!
//! # Element-specific functions
//!
//! Anything carrying a coefficient table goes through
//! [`SpecializedComplexSpecialMath`], which is implemented per element type the way
//! `thermite-special`'s own `ps.rs`/`pd.rs` are. [`trigamma`](SpecializedSpecialMath::trigamma)
//! and [`lambert_w`](SpecializedSpecialMath::lambert_w) are implemented there;
//! [`beta`](SpecializedSpecialMath::beta) is a default written on `tgamma`. So is the
//! Faddeeva function, whose own module is [`faddeeva`].
//!
//! The whole Gamma family is implemented, on the tables `thermite-special` exports
//! from [`thermite_special::tables::gamma`]. Only part of each table survives the crossing:
//! the Lanczos sums and the digamma `p_large` are analytic approximations that hold
//! off the real axis, while the digamma `[1, 2]` rational and the trigamma regions
//! are minimax fits to real intervals and are unusable here - so `digamma` and
//! `trigamma` lean on a recurrence where the real versions reach for a rational.
//!
//! [`expint`](SpecializedSpecialMath::expint) is inherited whole and simply runs in
//! complex arithmetic; what changes is [`ExpIntDetails`], all three methods of it.
//! [`use_series`](ExpIntDetails::use_series) picks the regime by `norm_sqr` rather than
//! the lexicographic `cmp_lt`, and additionally claims the whole left half-plane, where
//! the Stieltjes continued fraction degrades toward the cut but the series stops
//! alternating and converges cleanly. [`invalid`](ExpIntDetails::invalid) drops the real
//! version's `x < 0` hole, since the principal branch covers the cut plane; the cut
//! itself needs no handling, as all of the multivaluedness is the `-ln z` term and the
//! principal `ln` already carries it. [`cf_tiny`](ExpIntDetails::cf_tiny) backs the Lentz
//! sentinel off `MIN_POSITIVE`, which a complex reciprocal squares into zero.
//!
//! `E_1` holds machine precision over the cut plane. Higher orders come off the order
//! recurrence and lose roughly `|z|^(N-1)/(N-1)!`, matching what the real path considers
//! reliable - but the asymptotic series that path swaps in past its threshold has no
//! complex counterpart yet, so very large `|z|` at high `N` is not covered.
//!
//! # Bessel and Airy functions
//!
//! `I`, `K`, `J`, `Y`, `H^(1)` and `H^(2)` of complex argument at **real** order, and
//! `Ai`, `Ai'`, `Bi`, `Bi'`, over the whole plane, in [`bessel`]. `thermite-special`'s
//! real-order `I`/`K` kernel runs in complex arithmetic on the right half-plane (its
//! decisions restated through `BesselDetails`), and everything else is a rotation or
//! continuation of that, exactly as in Amos. `I`, `K`, `J`, `Y`, their scaled forms
//! (`bessel::<Scaled<J>>` is SciPy's `jve`) and the Airy entries are on
//! [`SpecialMath`](thermite_special::SpecialMath) under the marker spelling. The Hankel
//! pair, complex-valued even at real `z`, is `hankel::<H1>` / `hankel::<Scaled<H2>>` on
//! [`ComplexSpecialMath`]. Complex _order_ is not implemented: a
//! `Real` order with an imaginary part yields NaN in that lane.
//!
//! # Not implemented
//!
//! The const-order `bessel_i`/`bessel_j` entries and the spherical family are not yet
//! available over C. The runtime-order entries above cover the cylindrical family.
//!
//! `Complex<Compensated<..>>` gets the element-agnostic functions but `todo!()`s the
//! table-driven ones; `Complex<Dual<..>>` has all of them, and differentiates through
//! them, since the shared bodies are generic over [`RealValue`](crate::RealValue).

use thermite::math::policy::{DefaultPolicy, Policy};
use thermite::math::{CoreMathWithPolicy as _, TranscendentalMathWithPolicy as _};
use thermite::prelude::*;
use thermite_special::BesselOrder;
use thermite_special::specialized::{ExpIntDetails, SpecializedSpecialMath};
#[cfg(feature = "dual")]
use thermite_special::tables::primal::GammaPrimalTables;

use crate::Complex;
use crate::math::specialized::ComplexVector;
use crate::vector::RealFloatVector;
use thermite::math::PrimalProjection;

pub mod bessel;
mod erf;
pub mod faddeeva;
mod gamma;
mod lambert_w;
mod polylog;
mod zeta;

use self::bessel::real_order;
pub use self::bessel::{H1, H2, HankelFn};
use self::erf::erf_erfc_positive;
use self::gamma::polygamma::polygamma_impl;
use self::gamma::{digamma_impl, lgamma_impl, tgamma_impl, trigamma_impl};
use self::lambert_w::lambert_w_impl;
use self::zeta::zeta_impl;

// `E` is named rather than written `V::Element` because the self-referential form
// sends the trait solver into an overflow. The supertrait pins it regardless.
impl<E, V: RealFloatVector<Element = E>> SpecializedSpecialMath<Complex<E>> for Complex<V>
where
    Complex<V>: SpecializedComplexSpecialMath<Complex<E>>
        + GenericVector<Mask = V::Mask, Signed = V::Signed>
        + ComplexVector<Real = V>,
{
    type ExpIntDetails = Self;

    /// `$I_\nu(z)$` at real order on the right half-plane. The `Real` payload of a
    /// complex-typed [`BesselOrder`] is complex. Complex _order_ is not implemented, so a
    /// lane whose order has an imaginary part is NaN. See
    /// [`complex_bessel_iv`](SpecializedComplexSpecialMath::complex_bessel_iv).
    #[inline(always)]
    fn bessel_iv<P: Policy, const SCALED: bool>(self, order: BesselOrder<Self, Self::Signed>) -> Self {
        let (order, bad) = real_order::<V>(order);
        bad.select(Self::NAN, self.complex_bessel_iv::<P, SCALED>(order))
    }

    /// `$K_\nu(z)$` at real order on the right half-plane. See
    /// [`bessel_iv`](Self::bessel_iv).
    #[inline(always)]
    fn bessel_kv<P: Policy, const SCALED: bool>(self, order: BesselOrder<Self, Self::Signed>) -> Self {
        let (order, bad) = real_order::<V>(order);
        bad.select(Self::NAN, self.complex_bessel_kv::<P, SCALED>(order))
    }

    /// `$J_\nu(z)$` at real order over the whole plane, unscaled. It grows like
    /// `$e^{|\mathrm{Im}\,z|}$`. `bessel::<Scaled<J>>` (reaching
    /// [`bessel_jv_scaled`](Self::bessel_jv_scaled)) is the form that does not overflow, and
    /// the Hankel pair is the right basis off the
    /// axis. See [`complex_bessel_jyh`](SpecializedComplexSpecialMath::complex_bessel_jyh).
    #[inline(always)]
    fn bessel_jv<P: Policy>(self, order: BesselOrder<Self, Self::Signed>) -> Self {
        let (order, bad) = real_order::<V>(order);
        bad.select(Self::NAN, self.complex_bessel_jyh::<P, false>(order).0)
    }

    /// `$Y_\nu(z)$` at real order over the whole plane, unscaled. See
    /// [`bessel_jv`](Self::bessel_jv).
    #[inline(always)]
    fn bessel_yv<P: Policy>(self, order: BesselOrder<Self, Self::Signed>) -> Self {
        let (order, bad) = real_order::<V>(order);
        bad.select(Self::NAN, self.complex_bessel_jyh::<P, false>(order).1)
    }

    /// `$e^{-|\mathrm{Im}\,z|}J_\nu(z)$`, SciPy's `jve`: what `z.bessel(Scaled(J), order)`
    /// reaches. The upstream default is the unscaled value, which is right on the real axis
    /// only.
    #[inline(always)]
    fn bessel_jv_scaled<P: Policy>(self, order: BesselOrder<Self, Self::Signed>) -> Self {
        let (order, bad) = real_order::<V>(order);
        bad.select(Self::NAN, self.complex_bessel_jyh::<P, true>(order).0)
    }

    /// `$e^{-|\mathrm{Im}\,z|}Y_\nu(z)$`, SciPy's `yve`, for `z.bessel(Scaled(Y), order)`.
    #[inline(always)]
    fn bessel_yv_scaled<P: Policy>(self, order: BesselOrder<Self, Self::Signed>) -> Self {
        let (order, bad) = real_order::<V>(order);
        bad.select(Self::NAN, self.complex_bessel_jyh::<P, true>(order).1)
    }

    // The Airy family over C. Unlike the real line, the single-value entries are
    // projections of the tuple: the two sectors each run two Bessel passes regardless, and
    // trimming a pass per wanted output is queued rather than done.

    /// `$(\mathrm{Ai}, \mathrm{Ai}', \mathrm{Bi}, \mathrm{Bi}')$` at complex `z`, over the
    /// whole plane. See [`bessel::airy`].
    #[inline(always)]
    fn airy_tuple<P: Policy>(self) -> (Self, Self, Self, Self) {
        self.complex_airy::<P, false>()
    }

    /// SciPy's `airye`: `$e^{\zeta}\mathrm{Ai}$`, `$e^{\zeta}\mathrm{Ai}'$`,
    /// `$e^{-|\mathrm{Re}\,\zeta|}\mathrm{Bi}$`, `$e^{-|\mathrm{Re}\,\zeta|}\mathrm{Bi}'$`
    /// with the principal `$\zeta = \tfrac{2}{3}z^{3/2}$`, everywhere. Unlike the real
    /// line, which leaves the oscillating side unscaled. See [`bessel::airy`].
    #[inline(always)]
    fn airy_tuple_scaled<P: Policy>(self) -> (Self, Self, Self, Self) {
        self.complex_airy::<P, true>()
    }

    #[inline(always)]
    fn airy_ai<P: Policy>(self) -> Self {
        self.complex_airy::<P, false>().0
    }

    #[inline(always)]
    fn airy_ai_scaled<P: Policy>(self) -> Self {
        self.complex_airy::<P, true>().0
    }

    #[inline(always)]
    fn airy_bi<P: Policy>(self) -> Self {
        self.complex_airy::<P, false>().2
    }

    #[inline(always)]
    fn airy_bi_scaled<P: Policy>(self) -> Self {
        self.complex_airy::<P, true>().2
    }

    #[inline(always)]
    fn airy_ai_prime<P: Policy>(self) -> Self {
        self.complex_airy::<P, false>().1
    }

    #[inline(always)]
    fn airy_ai_prime_scaled<P: Policy>(self) -> Self {
        self.complex_airy::<P, true>().1
    }

    #[inline(always)]
    fn airy_bi_prime<P: Policy>(self) -> Self {
        self.complex_airy::<P, false>().3
    }

    #[inline(always)]
    fn airy_bi_prime_scaled<P: Policy>(self) -> Self {
        self.complex_airy::<P, true>().3
    }

    /// The error function over the whole complex plane.
    ///
    /// `erf` is entire and odd. The negative-real half-plane comes from
    /// `erf(-z) = -erf(z)`, a conditional negation, not a branch.
    #[inline(always)]
    fn erf<P: Policy>(self) -> Self {
        let neg = self.re.is_negative();
        let z = Complex::new(self.re.neg_c(neg), self.im.neg_c(neg));

        let (erf, _) = erf_erfc_positive::<P, E, V>(z);

        Complex::new(erf.re.neg_c(neg), erf.im.neg_c(neg))
    }

    /// The complementary error function over the whole complex plane.
    ///
    /// The `1 - erf(z)` default cancels for large `Re z`, where `erfc` is the function
    /// one wants in the first place; the continued fraction computes it directly there.
    /// The negative half-plane uses `erfc(z) = 2 - erfc(-z)`.
    #[inline(always)]
    fn erfc<P: Policy>(self) -> Self {
        let neg = self.re.is_negative();
        let z = Complex::new(self.re.neg_c(neg), self.im.neg_c(neg));

        let (_, erfc) = erf_erfc_positive::<P, E, V>(z);

        neg.select(Self::TWO - erfc, erfc)
    }

    /// `$\sigma(z) = \frac{1}{1 + e^{-z}}$`
    ///
    /// The default stabilizes for the real axis by negating on `is_positive()` and
    /// selecting, neither of which is holomorphic. The plain definition is, and the
    /// default itself falls back to it at lower precision policies.
    #[inline(always)]
    fn logistic_sigmoid<P: Policy>(self) -> Self {
        (Self::ONE + (-self).exp_p::<P>()).approx_reciprocal_p::<P>()
    }

    /// `$\frac{1}{k}\ln(1 + e^{kz})$`
    ///
    /// The default's `max(x, 0) + ln1p(e^{-|kx|})` is the real-axis overflow-stable
    /// rearrangement, and neither `|x|` nor `max` is holomorphic. This uses the
    /// analytic definition, and so overflows for large `Re(kz)` where the real form
    /// would not.
    #[inline(always)]
    fn softplus<P: Policy>(self, k: Self, rcp_k: Self) -> Self {
        (Self::ONE + (self * k).exp_p::<P>()).ln_p::<P>() * rcp_k
    }

    // The Gamma family and Lambert W carry element-specific coefficient tables, so
    // they route through `SpecializedComplexSpecialMath` below rather than living here.

    #[inline(always)]
    fn tgamma<P: Policy>(self) -> Self {
        self.complex_tgamma::<P>()
    }

    #[inline(always)]
    fn lgamma<P: Policy>(self) -> Self {
        self.complex_lgamma::<P>()
    }

    #[inline(always)]
    fn digamma<P: Policy>(self) -> Self {
        self.complex_digamma::<P>()
    }

    #[inline(always)]
    fn trigamma<P: Policy>(self) -> Self {
        self.complex_trigamma::<P>()
    }

    #[inline(always)]
    fn polygamma<P: Policy>(self, n: u32) -> Self {
        self.complex_polygamma::<P>(n)
    }

    #[inline(always)]
    fn zeta<P: Policy>(self) -> Self {
        self.complex_zeta::<P, false>()
    }

    #[inline(always)]
    fn zetac<P: Policy>(self) -> Self {
        self.complex_zeta::<P, true>()
    }

    /// `$\mathrm{Li}_s(z)$` over the whole plane at a real scalar order (given in this
    /// vector's complex element, and a non-real order answers NaN).
    #[inline(always)]
    fn polylog<P: Policy>(
        self,
        order: thermite_special::PolylogOrder<Complex<E>, <<Self as GenericVector>::Signed as GenericVector>::Element>,
    ) -> Self {
        self.complex_polylog::<P>(order)
    }

    #[inline(always)]
    fn beta<P: Policy>(a: Self, b: Self) -> Self {
        a.complex_beta::<P>(b)
    }

    /// Both `$W_0$` and `$W_{-1}$` are genuine branches of the complex Lambert W, so
    /// the real signature carries over unchanged - it simply cannot reach `$W_k$` for
    /// `$|k| \ge 2$`.
    #[inline(always)]
    fn lambert_w<P: Policy>(self) -> (Self, Self) {
        self.complex_lambert_w::<P>()
    }

    // TEMP(bessel_j): disabled until orders beyond J_0 exist - see thermite-special/src/lib.rs.
    //#[inline(always)]
    //fn bessel_j<P: Policy, const N: usize>(self) -> Self {
    //    todo!()
    //}
}

/// All three of the shared `expint` kernel's decisions change over C. Everything else
/// about that kernel - the series, the Lentz continued fraction, the order recurrence -
/// is inherited unchanged and simply runs in complex arithmetic.
impl<E, V: RealFloatVector<Element = E>> ExpIntDetails<Complex<E>, Complex<V>> for Complex<V>
where
    Complex<V>: FloatVector<Element = Complex<E>, Mask = V::Mask>,
{
    #[inline(always)]
    fn use_series(z: Complex<V>) -> V::Mask {
        // Two regions, for two different reasons.
        //
        // The unit disc is the real rule, but by modulus rather than by `cmp_lt` (which
        // on `Complex` is the lexicographic sort order, and would hand a point like
        // 0.5 + 100i to the series, where it diverges). Compared as `norm_sqr < 1` to
        // skip the root - squaring is monotone and the threshold is its own square.
        //
        // The whole left half-plane is added because the Stieltjes continued fraction
        // degrades as arg z approaches the cut - measurably by |Arg z| ~ 177 deg, and
        // completely on the cut itself. The series has no such trouble there: its terms
        // are (-z)^k / (k k!), so for Re z < 0 they stop alternating and the sum simply
        // accumulates, which is the cancellation-free direction. (The reverse of the
        // real line, where positive x is exactly what makes the series cancel and the
        // fraction is preferred.)
        z.norm_sqr().cmp_lt(V::ONE) | z.re.is_negative()
    }

    /// Nothing but NaN is out of domain.
    ///
    /// Real `E_N` is a half-line function and the default NaNs out `x < 0`. `E_N(z)` is
    /// holomorphic on the whole cut plane `|Arg z| < pi`, so the negative reals are
    /// in-domain here, approached from above. The cut needs no handling of its own: all
    /// of the multivaluedness sits in the `-ln z` term of the series, and the principal
    /// `ln` this crate provides already carries exactly that branch.
    #[inline(always)]
    fn invalid(z: Complex<V>) -> V::Mask {
        z.is_nan()
    }

    /// The default sentinel, `MIN_POSITIVE`, cannot be used here: a complex reciprocal
    /// is `conj(z) / |z|^2`, and `MIN_POSITIVE^2` underflows to zero, so the very first
    /// Lentz step divides by zero and every continued-fraction lane comes back NaN.
    ///
    /// `sqrt(MIN_POSITIVE) / EPSILON` is the principled choice: the square root is the
    /// hard floor for surviving the squaring, and dividing by EPSILON backs off it far
    /// enough that the reciprocal's square stays inside the exponent range too. Holds
    /// with room to spare for f32 and f64 alike.
    #[inline(always)]
    fn cf_tiny() -> Complex<V> {
        Complex::real(V::MIN_POSITIVE.sqrt() / <V as FloatVector>::EPSILON)
    }
}

// ---------------------------------------------------------------------------
// Per-element hook
// ---------------------------------------------------------------------------

/// The complex special functions whose algorithms carry element-specific coefficient
/// tables.
///
/// [`SpecializedSpecialMath`] for `Complex<V>` is one blanket impl that forwards here,
/// so the f32 and f64 cases can diverge exactly the way `thermite-special`'s own
/// `ps.rs`/`pd.rs` do. The shared bodies are free functions in this module that take
/// their coefficients as slices; an impl supplies the table and little else.
///
/// `Self` is the *complex* vector, so these are ordinary `self` methods and the
/// per-element dispatch is in the impl header (`for Complex<V> where V::Element = f64`).
/// That shape is what lets [`decl_complex_math!`](crate::math::specialized) generate
/// [`ComplexSpecialMath`] from it, exactly as it generates
/// [`ComplexMath`](crate::math::ComplexMath) from `SpecializedComplexMath`.
///
/// A type only reaches `SpecialMath` over C by implementing this, which is why the
/// element-agnostic members (`erf`, `logistic_sigmoid`, ...) are not on it: they stay
/// in the blanket impl and cost an implementor nothing. The Gamma members keep their
/// `complex_` prefix because `SpecialMath` already exposes `tgamma`/`lgamma`/... on the
/// same types, and two traits offering one name makes every call ambiguous.
pub trait SpecializedComplexSpecialMath<E>: ComplexVector<Element = E> {
    fn complex_tgamma<P: Policy>(self) -> Self;
    fn complex_lgamma<P: Policy>(self) -> Self;
    fn complex_digamma<P: Policy>(self) -> Self;
    fn complex_trigamma<P: Policy>(self) -> Self;

    /// `$\psi_n(z)$`, the n-th derivative of [`complex_digamma`](Self::complex_digamma).
    ///
    /// The default delegates the orders every implementor already carries (`n = 0`
    /// and `1`) and returns NaN above them. The f32/f64 impls override it with the
    /// full recurrence-plus-series kernel. The composite storage types inherit the
    /// default until they carry element tables of their own: for `Complex<Dual<..>>`
    /// that means `polygamma(n >= 2)` is NaN even though `trigamma` (which runs the
    /// shared body in dual arithmetic) differentiates fine. Closing that is queued.
    #[inline(always)]
    fn complex_polygamma<P: Policy>(self, n: u32) -> Self {
        match n {
            0 => self.complex_digamma::<P>(),
            1 => self.complex_trigamma::<P>(),
            // Not implemented for this storage type rather than undefined: `n` is a scalar,
            // so this is a whole-call decision and can be loud. Contrast the _per-lane_ NaNs
            // in the real kernel (reflected lanes past the cot table, orders past the
            // factorial table), which are limits of an implemented algorithm and cannot
            // panic: one lane of a vector has no way to.
            _ => todo!("complex polygamma(n >= 2) needs element tables this storage type lacks"),
        }
    }

    /// `$\zeta(z)$`, or `$\zeta(z) - 1$` when `ZETAC` is set.
    ///
    /// The same Euler-Maclaurin expansion the real kernel runs, in complex arithmetic: the
    /// tabulated pieces (the Bernoulli numbers, the base-2 logarithms of the primes under `N`)
    /// are properties of the _real_ element and carry over unchanged, which is why this needs
    /// an element-keyed impl rather than riding the blanket one.
    ///
    /// **Accuracy is governed by `|Im z|` against `N`.** The expansion's correction terms grow
    /// like `$(|z|/N)^{2k}$`, so it holds while `N` exceeds the imaginary part and degrades
    /// once it does not: measured on the critical line at `N = 10`, 4.7e-16 at `t = 4`, 2.3e-13
    /// at `t = 10`, 1.7e-9 at `t = 20`. Raising `N` tracks `t` linearly while the transcendental
    /// count only follows `$\pi(N)$` (`t = 20` wants `N = 28` and 9 exponentials, `t = 100`
    /// wants `N = 112` and 29), so the ceiling is a cost decision rather than an algorithmic
    /// one, up to the point where Riemann-Siegel takes over for zero-hunting at
    /// `$t \gtrsim 10^4$`.
    #[inline(always)]
    fn complex_zeta<P: Policy, const ZETAC: bool>(self) -> Self {
        todo!("complex zeta is not implemented for this storage type; the f32/f64 impls override it")
    }

    /// `$\mathrm{Li}_s(z)$` at a real scalar order, over the whole plane. The order is spelled
    /// in this vector's own element types (its complex element for `Real`, its signed lane
    /// element for `Integer`). The storage-agnostic default panics like `complex_zeta`.
    #[inline(always)]
    fn complex_polylog<P: Policy>(
        self,
        order: thermite_special::PolylogOrder<E, <<Self as GenericVector>::Signed as GenericVector>::Element>,
    ) -> Self {
        let _ = order;
        todo!("complex polylog is not implemented for this storage type; the f32/f64 impls override it")
    }

    fn complex_lambert_w<P: Policy>(self) -> (Self, Self);

    /// `$I_\nu(z)$` at **real** order and complex argument, or `$e^{-z}I_\nu(z)$` when
    /// `SCALED`, on the closed right half-plane `$\mathrm{Re}\,z \ge 0$`.
    #[inline(always)]
    fn complex_bessel_iv<P: Policy, const SCALED: bool>(
        self,
        _order: BesselOrder<Self::Real, <Self::Real as GenericVector>::Signed>,
    ) -> Self {
        todo!("complex bessel_iv needs the element tables this storage type lacks")
    }

    /// `$K_\nu(z)$` at real order, or `$e^{z}K_\nu(z)$` when `SCALED`. See
    /// [`complex_bessel_iv`](Self::complex_bessel_iv).
    #[inline(always)]
    fn complex_bessel_kv<P: Policy, const SCALED: bool>(
        self,
        _order: BesselOrder<Self::Real, <Self::Real as GenericVector>::Signed>,
    ) -> Self {
        todo!("complex bessel_kv needs the element tables this storage type lacks")
    }

    /// `$(J_\nu, Y_\nu, H^{(1)}_\nu, H^{(2)}_\nu)$` at real order over the whole plane, all
    /// from one `I`/`K` evaluation at a rotated argument. With `SCALED`, SciPy's `jve` /
    /// `yve` / `hankel1e` / `hankel2e` scalings.
    #[inline(always)]
    fn complex_bessel_jyh<P: Policy, const SCALED: bool>(
        self,
        _order: BesselOrder<Self::Real, <Self::Real as GenericVector>::Signed>,
    ) -> (Self, Self, Self, Self) {
        todo!("complex bessel_jv/yv/hankel need the element tables this storage type lacks")
    }

    /// `hankel::<W>(order)`: `$H^{(1)}_\nu$` or `$H^{(2)}_\nu$`, plain or scaled, selected
    /// by a [`HankelFn`] marker. See [`H1`].
    #[inline(always)]
    fn hankel<P: Policy, W: HankelFn>(
        self,
        order: BesselOrder<Self::Real, <Self::Real as GenericVector>::Signed>,
    ) -> Self {
        W::eval::<P, E, Self>(self, order)
    }

    /// `$(\mathrm{Ai}, \mathrm{Ai}', \mathrm{Bi}, \mathrm{Bi}')$` at complex `z`, scaled as
    /// SciPy's `airye` when `SCALED`. See [`bessel::airy`]. The default is for storage
    /// types without element tables.
    #[inline(always)]
    fn complex_airy<P: Policy, const SCALED: bool>(self) -> (Self, Self, Self, Self) {
        todo!("complex airy needs the element tables this storage type lacks")
    }

    /// The Faddeeva function `$w(z) = e^{-z^2}\operatorname{erfc}(-iz)$`.
    ///
    /// Carries the Weideman coefficient table, hence its place here. See
    /// [`faddeeva`] for the algorithm and the accuracy ladder.
    fn faddeeva_w<P: Policy>(self) -> Self;

    /// `$\operatorname{erfcx}(z) = e^{z^2}\operatorname{erfc}(z) = w(iz)$`.
    ///
    /// The scaled complementary error function: `erfc` without the exponential
    /// underflow, so it stays meaningful where `erfc` itself has flushed to zero.
    #[inline(always)]
    fn erfcx<P: Policy>(self) -> Self {
        // erfcx(z) = w(iz), and i*(x + iy) = -y + ix
        Self::from_parts(-self.im(), self.re()).faddeeva_w::<P>()
    }

    /// The Voigt function `$K(x, y) = \operatorname{Re} w(x + iy)$`, the convolution of
    /// a Gaussian and a Lorentzian in normalized coordinates, as a real value.
    ///
    /// The one consumer that wants the real part *alone*, and so the one that depends on
    /// the near-real-axis correction. Use `Best` or above; that is where it is enabled.
    ///
    /// Normalization is left to the caller: physical line shapes want an additional
    /// `$1/(\sigma\sqrt{2\pi})$` and a scaling of `x` and `y` by the Doppler width.
    #[inline(always)]
    fn voigt<P: Policy>(self) -> Self::Real {
        self.faddeeva_w::<P>().re()
    }

    /// `$B(a, b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}$`
    ///
    /// Deliberately *not* `exp(lgamma(a) + lgamma(b) - lgamma(a+b))`: over C the
    /// principal log-gamma branches do not add, so the exponentiated form is correct
    /// only up to a factor of `$e^{2\pi i k}$`. The quotient of gammas has no branch
    /// to get wrong, at the cost of overflowing where the log form would not.
    #[inline(always)]
    fn complex_beta<P: Policy>(self, b: Self) -> Self {
        self.complex_tgamma::<P>() * b.complex_tgamma::<P>() / (self + b).complex_tgamma::<P>()
    }
}

// ---------------------------------------------------------------------------
// Per-element impls
// ---------------------------------------------------------------------------

/// `$B_2, B_4, \ldots, B_{14}$`. Exact rationals, so f32 and f64 differ only in the
/// rounding; what differs materially is how many are worth keeping.
macro_rules! bernoulli_b2n {
    ($t:ty) => {
        [
            1.0 / 6.0,
            -1.0 / 30.0,
            1.0 / 42.0,
            -1.0 / 30.0,
            5.0 / 66.0,
            -691.0 / 2730.0,
            7.0 / 6.0,
        ]
    };
}

// The `Primal = V` pin: a real vector is its own primal, but the rigid
// `PrimalProjection` supertrait shadows the fixpoint blanket impl on a generic `V`, so
// without it `Complex<V>::Primal` will not normalize to `V` and the table lookup cannot
// resolve. Same pin thermite's `ps.rs`/`pd.rs` carry, for the same reason.
impl<V: RealFloatVector<Element = f32> + PrimalProjection<Primal = V> + FloatVectorWithBits>
    SpecializedComplexSpecialMath<Complex<f32>> for Complex<V>
{
    #[inline(always)]
    fn complex_tgamma<P: Policy>(self) -> Self {
        tgamma_impl::<P, V>(self)
    }

    #[inline(always)]
    fn complex_lgamma<P: Policy>(self) -> Self {
        lgamma_impl::<P, V>(self)
    }

    #[inline(always)]
    fn complex_digamma<P: Policy>(self) -> Self {
        digamma_impl::<P, V>(self)
    }

    #[inline(always)]
    fn complex_trigamma<P: Policy>(self) -> Self {
        // 24 bits of mantissa are exhausted long before the table is, so stop early
        // and let the (cheaper) recurrence make up the difference.
        const B: [f32; 4] = [1.0 / 6.0, -1.0 / 30.0, 1.0 / 42.0, -1.0 / 30.0];
        trigamma_impl::<P, V, 4>(self, &B, 8.0)
    }

    #[inline(always)]
    fn complex_zeta<P: Policy, const ZETAC: bool>(self) -> Self {
        zeta_impl::<P, V, ZETAC>(self)
    }

    #[inline(always)]
    fn complex_polylog<P: Policy>(self, order: thermite_special::PolylogOrder<Complex<f32>, i32>) -> Self {
        polylog::polylog_impl::<P, V, i32>(self, order)
    }

    #[inline(always)]
    fn complex_polygamma<P: Policy>(self, n: u32) -> Self {
        match n {
            0 => self.complex_digamma::<P>(),
            1 => self.complex_trigamma::<P>(),
            // Same shift base as `complex_trigamma`. The kernel adds its own 4n.
            _ => polygamma_impl::<P, V>(self, n, 8),
        }
    }

    #[inline(always)]
    fn complex_lambert_w<P: Policy>(self) -> (Self, Self) {
        const C: [f32; 2] = [11.0 / 72.0, 0.09];
        lambert_w_impl::<P, V>(self, &C, 3)
    }

    #[inline(always)]
    fn faddeeva_w<P: Policy>(self) -> Self {
        self::faddeeva::faddeeva_w::<P, f32, f32, V>(self)
    }

    #[inline(always)]
    fn complex_bessel_iv<P: Policy, const SCALED: bool>(self, order: BesselOrder<V, V::Signed>) -> Self {
        bessel::ik_whole_plane::<P, f32, V, 11, 11, SCALED, true>(
            order.simplify().to_real(),
            self,
            &thermite_special::tables::lgamma1p::LGAMMA1P_F32,
            thermite_special::tables::bessel::BESSEL_I0_F32.far_threshold,
        )
        .0
    }

    #[inline(always)]
    fn complex_bessel_kv<P: Policy, const SCALED: bool>(self, order: BesselOrder<V, V::Signed>) -> Self {
        bessel::ik_whole_plane::<P, f32, V, 11, 11, SCALED, false>(
            order.simplify().to_real(),
            self,
            &thermite_special::tables::lgamma1p::LGAMMA1P_F32,
            thermite_special::tables::bessel::BESSEL_I0_F32.far_threshold,
        )
        .1
    }

    #[inline(always)]
    fn complex_bessel_jyh<P: Policy, const SCALED: bool>(
        self,
        order: BesselOrder<V, V::Signed>,
    ) -> (Self, Self, Self, Self) {
        bessel::jyh_whole_plane::<P, f32, V, 11, 11, SCALED>(
            order.simplify().to_real(),
            self,
            &thermite_special::tables::lgamma1p::LGAMMA1P_F32,
            thermite_special::tables::bessel::BESSEL_I0_F32.far_threshold,
        )
    }

    #[inline(always)]
    fn complex_airy<P: Policy, const SCALED: bool>(self) -> (Self, Self, Self, Self) {
        bessel::airy::airy_whole_plane::<P, f32, V, 11, 11, SCALED>(
            self,
            &thermite_special::tables::lgamma1p::LGAMMA1P_F32,
            &thermite_special::tables::bessel::airy::AIRY_ZERO_F32,
            thermite_special::tables::bessel::BESSEL_I0_F32.far_threshold,
        )
    }
}

// See the `Primal = V` pin note on the f32 impl above.
impl<V: RealFloatVector<Element = f64> + PrimalProjection<Primal = V> + FloatVectorWithBits>
    SpecializedComplexSpecialMath<Complex<f64>> for Complex<V>
{
    #[inline(always)]
    fn complex_tgamma<P: Policy>(self) -> Self {
        tgamma_impl::<P, V>(self)
    }

    #[inline(always)]
    fn complex_lgamma<P: Policy>(self) -> Self {
        lgamma_impl::<P, V>(self)
    }

    #[inline(always)]
    fn complex_digamma<P: Policy>(self) -> Self {
        digamma_impl::<P, V>(self)
    }

    #[inline(always)]
    fn complex_trigamma<P: Policy>(self) -> Self {
        const B: [f64; 7] = bernoulli_b2n!(f64);
        trigamma_impl::<P, V, 7>(self, &B, 16.0)
    }

    #[inline(always)]
    fn complex_zeta<P: Policy, const ZETAC: bool>(self) -> Self {
        zeta_impl::<P, V, ZETAC>(self)
    }

    #[inline(always)]
    fn complex_polylog<P: Policy>(self, order: thermite_special::PolylogOrder<Complex<f64>, i64>) -> Self {
        polylog::polylog_impl::<P, V, i64>(self, order)
    }

    #[inline(always)]
    fn complex_polygamma<P: Policy>(self, n: u32) -> Self {
        match n {
            0 => self.complex_digamma::<P>(),
            1 => self.complex_trigamma::<P>(),
            // Same shift base as `complex_trigamma`. The kernel adds its own 4n.
            _ => polygamma_impl::<P, V>(self, n, 16),
        }
    }

    #[inline(always)]
    fn complex_lambert_w<P: Policy>(self) -> (Self, Self) {
        const C: [f64; 2] = [11.0 / 72.0, 0.09];
        lambert_w_impl::<P, V>(self, &C, 4)
    }

    #[inline(always)]
    fn faddeeva_w<P: Policy>(self) -> Self {
        self::faddeeva::faddeeva_w::<P, f64, f64, V>(self)
    }

    #[inline(always)]
    fn complex_bessel_iv<P: Policy, const SCALED: bool>(self, order: BesselOrder<V, V::Signed>) -> Self {
        bessel::ik_whole_plane::<P, f64, V, 25, 25, SCALED, true>(
            order.simplify().to_real(),
            self,
            &thermite_special::tables::lgamma1p::LGAMMA1P_F64,
            thermite_special::tables::bessel::BESSEL_I0_F64.far_threshold,
        )
        .0
    }

    #[inline(always)]
    fn complex_bessel_kv<P: Policy, const SCALED: bool>(self, order: BesselOrder<V, V::Signed>) -> Self {
        // `NEED_I = false`: `K` is the cheap half and all this entry wants on the
        // right half-plane. The left needs `I` for the continuation and asks for it.
        bessel::ik_whole_plane::<P, f64, V, 25, 25, SCALED, false>(
            order.simplify().to_real(),
            self,
            &thermite_special::tables::lgamma1p::LGAMMA1P_F64,
            thermite_special::tables::bessel::BESSEL_I0_F64.far_threshold,
        )
        .1
    }

    #[inline(always)]
    fn complex_bessel_jyh<P: Policy, const SCALED: bool>(
        self,
        order: BesselOrder<V, V::Signed>,
    ) -> (Self, Self, Self, Self) {
        bessel::jyh_whole_plane::<P, f64, V, 25, 25, SCALED>(
            order.simplify().to_real(),
            self,
            &thermite_special::tables::lgamma1p::LGAMMA1P_F64,
            thermite_special::tables::bessel::BESSEL_I0_F64.far_threshold,
        )
    }

    #[inline(always)]
    fn complex_airy<P: Policy, const SCALED: bool>(self) -> (Self, Self, Self, Self) {
        bessel::airy::airy_whole_plane::<P, f64, V, 25, 25, SCALED>(
            self,
            &thermite_special::tables::lgamma1p::LGAMMA1P_F64,
            &thermite_special::tables::bessel::airy::AIRY_ZERO_F64,
            thermite_special::tables::bessel::BESSEL_I0_F64.far_threshold,
        )
    }
}

// The composite storage types reach `SpecialMath` over C through this trait like any
// other. They take no position on the coefficient tables - which is the status quo,
// since the Gamma family was unimplemented for them before this trait existed too -
// but implementing it is what keeps `erf`, `erfc`, `logistic_sigmoid` and the
// polynomial families working on `Complex<Dual<..>>` / `Complex<Compensated<..>>`.
//
// The element type is written out structurally rather than as an associated type so
// that it is visibly disjoint from the f32/f64 impls above; a bare `E` would leave
// coherence unable to prove the two cannot overlap.
#[cfg(feature = "dual")]
impl<E, V: FloatVector<Element = E>, const N: usize> SpecializedComplexSpecialMath<Complex<thermite_dual::Dual<E, N>>>
    for Complex<thermite_dual::Dual<V, N>>
where
    E: thermite::element::FloatElementWithBits + thermite_dual::DualValue + self::faddeeva::WeidemanTables,
    // `Dual<V, N>::Primal` is `V::Primal`, recursively, so a `Dual` over a real vector
    // reaches that vector's own table. Stated as an equality rather than left to
    // normalize: selecting thermite-dual's `PrimalProjection` impl would need
    // `V: DualMathVector`, which is more than this impl otherwise requires.
    //
    // This is also where the f32 inner stops paying for the f64 table: the provider is
    // selected on `V`, where the element is concrete, so an f32 inner supplies its own
    // 6-term Lanczos instead of the narrowed 13-term one the old adapter hardcoded.
    thermite_dual::Dual<V, N>: RealFloatVector<Element = thermite_dual::Dual<E, N>>
        + PrimalProjection<Primal = V>
        + thermite::math::specialized::SpecializedCoreMath<thermite_dual::Dual<E, N>>,
    V: GammaPrimalTables<E>,
{
    #[inline(always)]
    fn complex_tgamma<P: Policy>(self) -> Self {
        tgamma_impl::<P, thermite_dual::Dual<V, N>>(self)
    }

    #[inline(always)]
    fn complex_lgamma<P: Policy>(self) -> Self {
        lgamma_impl::<P, thermite_dual::Dual<V, N>>(self)
    }

    #[inline(always)]
    fn complex_digamma<P: Policy>(self) -> Self {
        digamma_impl::<P, thermite_dual::Dual<V, N>>(self)
    }

    /// The shared body is already generic over `RealValue`, so it differentiates
    /// itself; all `Dual` has to supply is the same table with zero derivative parts,
    /// which is what `Dual::constant` means.
    ///
    /// This is `psi_2` by forward-mode AD, without a tetragamma ever being written.
    #[inline(always)]
    fn complex_trigamma<P: Policy>(self) -> Self {
        // The f64-grade table regardless of the inner element: `V::Element` is not
        // known here, and over-converging an f32 costs a few terms rather than
        // correctness.
        let b = [
            thermite_dual::Dual::<E, N>::constant(E::from_f64(1.0 / 6.0)),
            thermite_dual::Dual::<E, N>::constant(E::from_f64(-1.0 / 30.0)),
            thermite_dual::Dual::<E, N>::constant(E::from_f64(1.0 / 42.0)),
            thermite_dual::Dual::<E, N>::constant(E::from_f64(-1.0 / 30.0)),
            thermite_dual::Dual::<E, N>::constant(E::from_f64(5.0 / 66.0)),
            thermite_dual::Dual::<E, N>::constant(E::from_f64(-691.0 / 2730.0)),
            thermite_dual::Dual::<E, N>::constant(E::from_f64(7.0 / 6.0)),
        ];

        trigamma_impl::<P, thermite_dual::Dual<V, N>, 7>(
            self,
            &b,
            thermite_dual::Dual::<E, N>::constant(E::from_f64(16.0)),
        )
    }

    #[inline(always)]
    fn complex_lambert_w<P: Policy>(self) -> (Self, Self) {
        let c = [
            thermite_dual::Dual::<E, N>::constant(E::from_f64(11.0 / 72.0)),
            thermite_dual::Dual::<E, N>::constant(E::from_f64(0.09)),
        ];

        lambert_w_impl::<P, thermite_dual::Dual<V, N>>(self, &c, 4)
    }

    #[inline(always)]
    fn faddeeva_w<P: Policy>(self) -> Self {
        use self::faddeeva::{Weideman, WeidemanTables, faddeeva_w_with, weideman_n};

        // The coefficient table and its matched `L` come from `E`, the *primal*
        // element, so an f32 inner gets f32's own Weideman constants and ladder rather
        // than f64's narrowed - the same fix the Gamma family got from
        // `GammaPrimalTables`. They also stay real: `horner_real` adds them through
        // `nmul_add_primal`, which touches the value component only.
        macro_rules! tier {
            ($n:literal) => {
                faddeeva_w_with::<P, thermite_dual::Dual<E, N>, E, thermite_dual::Dual<V, N>, $n>(
                    self,
                    <E as Weideman<$n>>::L,
                    &<E as Weideman<$n>>::A,
                    thermite_dual::Dual::constant(<E as WeidemanTables>::HUGE),
                    thermite_dual::Dual::constant(<E as WeidemanTables>::REAL_AXIS_Y),
                    thermite_dual::Dual::constant(<E as WeidemanTables>::REAL_AXIS_X),
                )
            };
        }

        macro_rules! is {
            ($n:literal) => {
                const { weideman_n(P::POLICY.precision, <E as WeidemanTables>::MAX_N) <= $n }
            };
        }

        if is!(8) {
            tier!(8)
        } else if is!(16) {
            tier!(16)
        } else if is!(24) {
            tier!(24)
        } else if is!(32) {
            tier!(32)
        } else {
            tier!(40)
        }
    }
}

/*
#[cfg(feature = "compensated")]
impl<V: FloatVector>
    SpecializedComplexSpecialMath<Complex<thermite_compensated::Compensated<V::Element>>>
    for Complex<thermite_compensated::Compensated<V>>
where
    thermite_compensated::Compensated<V>: RealFloatVector<Element = thermite_compensated::Compensated<V::Element>>,
{
    #[inline(always)]
    fn complex_tgamma<P: Policy>(self) -> Self {
        todo!("complex tgamma over Compensated: needs the real complex tgamma first")
    }

    #[inline(always)]
    fn complex_lgamma<P: Policy>(self) -> Self {
        todo!("complex lgamma over Compensated: needs the real complex lgamma first")
    }

    #[inline(always)]
    fn complex_digamma<P: Policy>(self) -> Self {
        todo!("complex digamma over Compensated: needs the real complex digamma first")
    }

    #[inline(always)]
    fn complex_trigamma<P: Policy>(self) -> Self {
        todo!("complex trigamma over Compensated: not yet ported - needs the table as Compensated constants")
    }

    #[inline(always)]
    fn complex_lambert_w<P: Policy>(self) -> (Self, Self) {
        todo!("complex lambert_w over Compensated: not yet ported - needs the coefficients as Compensated constants")
    }

    #[inline(always)]
    fn faddeeva_w<P: Policy>(self) -> Self {
        todo!("Faddeeva over Compensated: needs Weideman tables as Compensated constants")
    }
}
*/

thermite::math_traits! {
    #![thermite(thermite)]
    #![surface(__complex_special_math_surface)]
    #![specialized(self)]

    /// Complex special functions that carry an element-specific coefficient table.
    ///
    /// Generated from [`SpecializedComplexSpecialMath`] by the same macro that builds
    /// [`ComplexMath`](crate::math::ComplexMath), so `z.faddeeva_w_p::<Precision>()` behaves as
    /// `z.norm_p::<Precision>()` does and generic code can bound on `V: ComplexSpecialMath`.
    ///
    /// Only the members with no home in [`SpecialMath`](thermite_special::SpecialMath) are
    /// re-exposed here. The Gamma family is already `z.tgamma()` there; declaring it a
    /// second time would make every such call ambiguous whenever both traits are in scope.
    #[element(FloatElement)]
    pub trait ComplexSpecialMath: ComplexVector {
        /// `$w(z) = e^{-z^2}\operatorname{erfc}(-iz)$`, the Faddeeva function (also the
        /// complex error function, or the plasma dispersion function up to a factor).
        ///
        /// See [`faddeeva`] for the algorithm, the accuracy ladder, and the one
        /// caveat that matters (`$\operatorname{Re} w$` near the real axis).
        fn faddeeva_w(self) -> Self;

        /// `$\operatorname{erfcx}(z) = e^{z^2}\operatorname{erfc}(z) = w(iz)$`, the
        /// scaled complementary error function.
        fn erfcx(self) -> Self;

        /// The Voigt function `$K(x, y) = \operatorname{Re} w(x + iy)$`, as a real value.
        fn voigt(self) -> Self::Real;

        /// A Hankel function at real order, selected by marker: [`H1`] for `$H^{(1)}_\nu$`,
        /// [`H2`] for `$H^{(2)}_\nu$`, or either under
        /// [`Scaled`](thermite_special::bessel::Scaled) for SciPy's `hankel1e`/`hankel2e`
        /// (`$e^{\mp iz}H^{(1,2)}_\nu$`).
        ///
        /// The Hankel functions are the oscillating basis to use off the real axis:
        /// `$H^{(1)}$` decays in the upper half-plane and `$H^{(2)}$` in the lower, where
        /// `$J$` and `$Y$` both grow like `$e^{|\mathrm{Im}\,z|}$`. They are complex-valued
        /// at real `z` too, which is why this lives here and not on
        /// [`SpecialMath`](thermite_special::SpecialMath), where `$J$`, `$Y$`, `$I$`, `$K$`
        /// and their scaled forms (`bessel::<Scaled<J>>`, SciPy's `jve`) are.
        ///
        /// ```rust,ignore
        /// let h1 = z.hankel::<H1>(BesselOrder::Real(nu));
        /// let h2e = z.hankel::<Scaled<H2>>(BesselOrder::Real(nu));
        /// ```
        fn hankel<W: HankelFn>(self, order: BesselOrder<Self::Real, <Self::Real as GenericVector>::Signed>) -> Self;
    }
}
