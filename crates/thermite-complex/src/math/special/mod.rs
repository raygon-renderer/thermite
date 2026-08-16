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
//! from [`thermite_special::tables`]. Only part of each table survives the crossing:
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
//! # Not implemented
//!
//! `bessel_j` is disabled crate-wide until orders beyond `J_0` exist upstream.
//!
//! `Complex<Compensated<..>>` gets the element-agnostic functions but `todo!()`s the
//! table-driven ones; `Complex<Dual<..>>` has all of them, and differentiates through
//! them, since the shared bodies are generic over [`RealValue`](crate::RealValue).

use thermite::math::policy::{DefaultPolicy, Policy};
use thermite::math::{CoreMathWithPolicy as _, FloatConsts, TranscendentalMathWithPolicy as _};
use thermite::prelude::*;
use thermite_special::specialized::{ExpIntDetails, SpecializedSpecialMath};

use crate::Complex;
use crate::math::ComplexMathWithPolicy as _;
use crate::math::specialized::ComplexVector;
use crate::vector::RealFloatVector;
use thermite::math::PrimalProjection;
use thermite_special::primal_tables::GammaPrimalTables;

pub mod faddeeva;

/// Terms of the Taylor series, which needs roughly `2|z|^2` of them.
///
/// The loop exits once every lane has converged, after ~20 for typical arguments,
/// so this cap only costs the large-`|z|` lanes that need it.
const SERIES_TERMS: usize = 160;

/// `$|z|^2$` past which the series cannot converge within [`SERIES_TERMS`].
///
/// The series needs roughly `$2|z|^2$` terms, so 160 of them reach `$|z| \approx 8.9$`.
const SERIES_RADIUS_SQ: i64 = 64;

/// `$-Re(z^2)$` past which the series alternates badly enough to be abandoned.
///
/// Pulling `$e^{-z^2}$` out front removes the alternation **only where `$Re(z^2) > 0$`**.
/// Near the imaginary axis `$z^2$` is negative real, the ratio `$2z^2/(2n+1)$` is
/// negative again, and the terms peak around `$e^{|z|^2}$` on their way to a sum of
/// order `$|z|$` - about `$0.43|z|^2$` digits lost. At `$z = 0.01 + 6i$` that is 15
/// digits, and the old implementation returned one correct digit there.
///
/// 8 keeps the loss under ~3.5 digits. Inside this band the series is the better regime
/// at every policy and is *structurally* exact in ways `w` is not: for purely imaginary
/// `z` every term is purely imaginary, so `erf(iy)` has a real part of exactly zero.
const SERIES_ALTERNATION_LIMIT: i64 = 8;

/// `$Re(z^2) = x^2 - y^2$` past which `erfc` must be computed directly rather than as
/// `1 - erf`.
///
/// `Re(z^2)`, not `|z|`, governs `$e^{-z^2}$`, hence how small `erfc` is, hence how
/// badly `1 - erf` cancels. By `x = 6` there is nothing left: `erf(6)` has rounded to
/// exactly 1.0 while `erfc(6)` is 2e-17.
const DIRECT_ERFC_LIMIT: i64 = 6;

/// `(erf(z), erfc(z))` for `Re z >= 0`. The reflections in the impl below need no more.
///
/// Both regimes are evaluated on every lane and blended, the lanes of a vector not
/// agreeing on which applies.
///
/// # Accuracy
///
/// ~1 ulp over the whole half-plane. `erfc` comes from [`faddeeva`] via
/// `$\operatorname{erfc}(z) = e^{-z^2}w(iz)$` for `$|z| \ge 1$` and from the Taylor
/// series inside that, and the two regimes have no gap between them.
///
/// This replaced a continued fraction keyed on `Re(z^2) >= 6`, which left a wedge of
/// large `|Im z|` where neither regime was valid: the series truncated before converging
/// and the continued fraction was never selected. Measured against a 50-digit oracle,
/// `erfc(0.1 + 10i)` was wrong by 36 orders of magnitude and is now good to 1e-13.
#[inline(always)]
fn erf_erfc_positive<P: Policy, E, V>(z: Complex<V>) -> (Complex<V>, Complex<V>)
where
    V: RealFloatVector<Element = E>,
    Complex<V>: SpecializedComplexSpecialMath<Complex<E>> + GenericVector<Mask = V::Mask>,
{
    let one = Complex::<V>::ONE;

    let z2 = z.square();
    let exp_nz2 = (-z2).exp_p::<P>(); // e^{-z^2}, common to both regimes

    // --- regime selection ---
    //
    // Three independent reasons to take `erfc` from `w` rather than from `1 - erf`:
    //
    //  1. `Re(z^2) >= 6`, where `erfc` is so much smaller than `erf` that the
    //     subtraction has no digits left. This was the old continued fraction's job.
    //  2. `Re(z^2) <= -8`, where the series alternates and cancels.
    //  3. `|z| > 8`, where the series truncates before it converges.
    //
    // Cases 2 and 3 together are the wedge, and neither was covered before: the
    // continued fraction was selected on `Re(z^2) >= 6`, so everything with large
    // `|Im z|` fell through to a series that could not deliver it.
    //
    // Inside the remaining band the series wins at every policy and is kept - `w` at
    // `Average` is 4e-10 where the series is at the last ulp. Widening this trades
    // accuracy in the bulk for nothing.
    let cancels = z2
        .re
        .cmp_ge(thermite::const_splat!(int <V::Element>: DIRECT_ERFC_LIMIT));
    let alternates = z2
        .re
        .cmp_le(thermite::const_splat!(int <V::Element>: -SERIES_ALTERNATION_LIMIT));
    let beyond_series = z
        .norm_sqr()
        .cmp_gt(thermite::const_splat!(int <V::Element>: SERIES_RADIUS_SQ));

    let use_w = cancels | alternates | beyond_series;

    let mut series_erf = Complex::<V>::EMPTY;
    let mut w_erfc = Complex::<V>::EMPTY;

    // --- Taylor series (Abramowitz & Stegun 7.1.6) ---
    //
    //   erf(z) = (2/sqrt(pi)) e^{-z^2} * sum_{n>=0} t_n,
    //     t_0 = z,  t_n = t_{n-1} * 2z^2 / (2n + 1)
    //
    // The e^{-z^2} factor absorbs the alternation where `Re(z^2) > 0`; case 2 above is
    // where it does not, and those lanes are on `w` instead.
    if const { P::POLICY.avoid_branching } || !use_w.all() {
        let two_z2 = z2 + z2;

        let mut term = z;
        let mut sum = z;

        let eps_sqr: V = <V as FloatVector>::EPSILON * <V as FloatVector>::EPSILON;

        let mut n = 1usize;
        while n < SERIES_TERMS {
            let denom = V::splat(<V::Element as FloatElement>::from_int(2 * n as thermite::LargeInt + 1));

            // `(term * two_z2) / denom` costs the same roundings, but puts the reciprocal
            // on the loop-carried chain: `term` then waits a division *and* a multiply
            // per iteration. Scaling the loop-invariant `two_z2` instead leaves one
            // complex multiply between successive terms, and the reciprocal issues
            // alongside it.
            let ratio = two_z2 / denom;

            term *= ratio;
            sum += term;

            // |term| <= eps * |sum| in every lane that will actually use this. The
            // `| use_w` matters: a single large-|z| lane never converges, and without it
            // one such lane drags the whole vector to SERIES_TERMS for a result that is
            // then discarded.
            let converged = term.norm_sqr().cmp_le(sum.norm_sqr() * eps_sqr);

            if (converged | use_w).all() {
                break;
            }

            n += 1;
        }

        series_erf = sum * exp_nz2 * <V as thermite::math::FloatConsts>::FRAC_2_SQRT_PI;
    }

    // --- Faddeeva, for the erfc side ---
    //
    //   erfc(z) = e^{-z^2} w(iz)
    //
    // `Re z >= 0` here, so `Im(iz) >= 0` and `w` never pays for its lower half-plane
    // reflection. `e^{-z^2}` is already in hand for the series, so this costs one
    // `w` - a single reciprocal and an N-term Horner - where the continued fraction it
    // replaced took 32 complex divisions.
    if const { P::POLICY.avoid_branching } || use_w.any() {
        w_erfc = exp_nz2 * SpecializedComplexSpecialMath::faddeeva_w::<P>(Complex::new(-z.im, z.re));
    }

    let erf = use_w.select(one - w_erfc, series_erf);
    let erfc = use_w.select(w_erfc, one - series_erf);

    (erf, erfc)
}

// `E` is named rather than written `V::Element` because the self-referential form
// sends the trait solver into an overflow; the supertrait pins it regardless.
impl<E, V: RealFloatVector<Element = E>> SpecializedSpecialMath<Complex<E>> for Complex<V>
where
    Complex<V>: SpecializedComplexSpecialMath<Complex<E>> + GenericVector<Mask = V::Mask>,
{
    type ExpIntDetails = Self;

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
        (Self::ONE + (-self).exp_p::<P>()).reciprocal_p::<P>()
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
    fn complex_lambert_w<P: Policy>(self) -> (Self, Self);

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
// Shared bodies
// ---------------------------------------------------------------------------

/// Shared complex `tgamma`, by the Lanczos approximation.
///
/// Lanczos is an analytic approximation, not a minimax fit, so it carries over from
/// the real implementation unchanged apart from the arithmetic. The left half-plane
/// comes from the reflection formula `Gamma(z)Gamma(1-z) = pi/sin(pi z)`.
///
/// Unlike the real version this does not split the `pow` in two for large arguments,
/// so it overflows around `Re z ~ 171` (f64) rather than reaching the very top of the
/// range. It also has no integer fast path: over C that test would only fire on a
/// measure-zero set.
#[inline(always)]
fn tgamma_impl<P: Policy, V: RealFloatVector>(z: Complex<V>) -> Complex<V>
where
    V::Primal: GammaPrimalTables<<V::Primal as GenericVector>::Element>,
{
    let l = <V::Primal as GammaPrimalTables<_>>::lanczos_primal();

    let reflect = z.re.cmp_lt(V::HALF);
    let w = reflect.select(Complex::ONE - z, z);

    let gh = V::from_primal(l.g) - V::HALF;
    let zgh = Complex::new(w.re + gh, w.im);

    let lanczos = w.poly_rev_primal_p::<P, _>(&l.p_rev) / w.poly_rev_primal_p::<P, _>(&l.q_rev);

    // zgh^(w - 1/2) * e^(-zgh) * lanczos_sum(w), with the two exponentials folded into
    // one: exp((w - 1/2) ln(zgh) - zgh).
    //
    // The real version cannot do this - it calls `powf` and then divides by `exp(zgh)`,
    // so the `pow` overflows on its own well before the product does, and it has to
    // split the exponent in half and square the result to compensate. Folding removes
    // the intermediate entirely: nothing overflows until the answer does. It is also
    // one transcendental cheaper, `powf` being `exp(e ln x)` underneath.
    //
    // `Re zgh >= g > 0` on this branch, so the `ln` never approaches its cut.
    let e = Complex::new(w.re - V::HALF, w.im);
    let res = (e * zgh.ln_p::<P>() - zgh).exp_p::<P>() * lanczos;

    // Gamma(z) = pi / (sin(pi z) Gamma(1 - z))
    let refl = Complex::real(<V as FloatConsts>::PI) / (z.sin_pi_p::<P>() * res);

    reflect.select(refl, res)
}

/// Shared complex `lgamma`, from the `exp(g)`-scaled Lanczos sum.
///
/// # Branch
///
/// For `Re z >= 1/2` this is the *continuous* log-gamma, not merely a principal
/// value: both logarithms it takes are of arguments confined to the right half-plane,
/// so neither crosses the cut, and the large imaginary parts come from the
/// `(z - 1/2) ln(zgh)` product rather than from a wrapped logarithm.
///
/// The reflected half-plane is another matter - `ln(sin(pi z))` is principal there, so
/// the result can differ from the continuous branch by a multiple of `2 pi i`.
#[inline(always)]
fn lgamma_impl<P: Policy, V: RealFloatVector>(z: Complex<V>) -> Complex<V>
where
    V::Primal: GammaPrimalTables<<V::Primal as GenericVector>::Element>,
{
    let l = <V::Primal as GammaPrimalTables<_>>::lanczos_primal();

    let reflect = z.re.cmp_lt(V::HALF);
    let w = reflect.select(Complex::ONE - z, z);

    let b = Complex::new(w.re - V::HALF, w.im);
    let a = Complex::new(b.re + V::from_primal(l.g), b.im).ln_p::<P>() - Complex::ONE;

    let s = w.poly_primal_p::<P, _>(&l.p_expg_scaled) / w.poly_primal_p::<P, _>(&l.q);

    let res = a * b + s.ln_p::<P>();

    // ln Gamma(z) = ln(pi) - ln(sin(pi z)) - ln Gamma(1 - z)
    let refl = Complex::real(<V as FloatConsts>::LN_PI) - z.sin_pi_p::<P>().ln_p::<P>() - res;

    reflect.select(refl, res)
}

/// Shared complex `digamma`.
///
/// Only `p_large` of the real [`Digamma`] table is usable here: the `[1, 2]` rational
/// beside it is a minimax fit to a real interval and says nothing off the axis, while
/// `p_large` is a genuine asymptotic series. So the recurrence does the work the
/// rational does in the real version, walking `Re z` up to `shift` before expanding.
#[inline(always)]
fn digamma_impl<P: Policy, V: RealFloatVector>(z: Complex<V>) -> Complex<V>
where
    V::Primal: GammaPrimalTables<<V::Primal as GenericVector>::Element>,
{
    let p_large = <V::Primal as GammaPrimalTables<_>>::digamma_p_large();

    let reflect = z.re.cmp_lt(V::HALF);

    let mut w = reflect.select(Complex::ONE - z, z);
    let mut refl = Complex::<V>::ZERO;

    if const { P::POLICY.avoid_branching } || reflect.any() {
        // psi(z) = psi(1 - z) - pi cot(pi z)
        let (s, c) = z.sincos_pi_p::<P>();
        refl = -(c / s * Complex::real(<V as FloatConsts>::PI));
    }

    // psi(w) = psi(w + 1) - 1/w, walked until the series below applies.
    let shift = V::from_primal(<V::Primal as GammaPrimalTables<_>>::digamma_shift());
    let mut acc = Complex::<V>::ZERO;
    let mut active = w.re.cmp_lt(shift);

    while active.any() {
        acc = active.select(acc - w.finv_p::<P>(), acc);
        w = active.select(w + Complex::ONE, w);
        active = w.re.cmp_lt(shift);
    }

    // psi(w) ~ ln(w-1) + 1/(2(w-1)) - u P(u),  u = 1/(w-1)^2
    let xm1 = w - Complex::ONE;
    let u = (xm1 * xm1).finv_p::<P>();

    let psi = xm1.ln_p::<P>() + (xm1 + xm1).finv_p::<P>() - u * u.poly_primal_p::<P, _>(&p_large);

    let total = acc + psi;

    reflect.select(refl + total, total)
}

/// Shared complex trigamma.
///
/// Three stages, none of them the real implementation's: that one leans on minimax
/// rationals fitted to intervals of the real line, which say nothing off the axis.
/// This is the classical route instead - reflect, recurse, expand.
///
/// * `bernoulli` are `$B_2, B_4, \ldots$` in order, the asymptotic series coefficients.
/// * `shift` is the `Re z` the recurrence walks up to before that series is used.
///   Both are the per-element tuning knobs: a shorter table wants a larger shift.
#[inline(always)]
fn trigamma_impl<P: Policy, V: RealFloatVector, const NB: usize>(
    z: Complex<V>,
    bernoulli: &[V::Element; NB],
    shift: V::Element,
) -> Complex<V> {
    // Reflect the left half-plane: psi_1(z) + psi_1(1 - z) = pi^2 / sin^2(pi z).
    let reflect = z.re.cmp_lt(V::HALF);

    let mut w = reflect.select(Complex::ONE - z, z);
    let mut refl = Complex::<V>::ZERO;

    if const { P::POLICY.avoid_branching } || reflect.any() {
        let s = z.sin_pi_p::<P>();
        refl = Complex::real(<V as FloatConsts>::PI_SQUARED) / (s * s);
    }

    // psi_1(w) = 1/w^2 + psi_1(w + 1), walked until Re w is large enough for the
    // series below. Bounded: the reflection already put Re w >= 1/2, so this runs at
    // most `shift` times.
    let shift = V::splat(shift);
    let mut acc = Complex::<V>::ZERO;
    let mut active = w.re.cmp_lt(shift);

    while active.any() {
        let t = (w * w).finv_p::<P>();
        acc = active.select(acc + t, acc);
        w = active.select(w + Complex::ONE, w);
        active = w.re.cmp_lt(shift);
    }

    // psi_1(w) ~ 1/w + 1/(2w^2) + sum_k B_2k / w^(2k+1)
    let u = w.finv_p::<P>();
    let u2 = u * u;

    // Horner in u^2, leading term first. The coefficients are real, so each step is
    // two FMAs on the real part (the coefficient rides the second) and two on the
    // imaginary - never a complex multiply followed by a separate add.
    let mut tail = Complex::real(V::splat(bernoulli[NB - 1]));
    let mut i = NB - 1;
    while i > 0 {
        i -= 1;

        let c = V::splat(bernoulli[i]);

        tail = Complex::new(
            tail.im.nmul_adde(u2.im, tail.re.mul_adde(u2.re, c)),
            tail.re.mul_adde(u2.im, tail.im * u2.re),
        );
    }

    let psi = acc + u + u2 * V::HALF + (u2 * u) * tail;

    reflect.select(refl - psi, psi)
}

/// Shared complex Lambert W, returning `$(W_0, W_{-1})$`.
///
/// Both are ordinary branches of `$W$` over C - unlike the real case, where `$W_{-1}$`
/// exists only on `$[-1/e, 0)$`. A Halley iteration from a per-region initial guess;
/// Halley is cubic, so a handful of steps suffices once the guess is in the right
/// basin, and picking that basin is the whole difficulty.
///
/// * `c` is `[11/72, r^2]`: the third Puiseux coefficient, and the squared radius
///   around `$-1/e$` inside which that series is used.
/// * `iters` is the Halley count.
///
/// Accurate away from the branch cuts. Near the cut on `$(-\infty, -1/e)$` the two
/// branches exchange values and a guess can land in the wrong basin, so results there
/// follow whichever branch the iteration converged to, not the principal labelling.
#[inline(always)]
fn lambert_w_impl<P: Policy, V: RealFloatVector>(
    z: Complex<V>,
    c: &[V::Element; 2],
    iters: usize,
) -> (Complex<V>, Complex<V>) {
    let e = <V as FloatConsts>::E;

    // --- Branch-point series in p = sqrt(2(ez + 1)) ---
    // The two branches meet at z = -1/e, where p = 0 and both equal -1:
    //   W_0    = -1 + p - p^2/3 + 11 p^3/72
    //   W_{-1} = -1 - p - p^2/3 - 11 p^3/72
    // The even term keeps its sign, the odd ones flip.
    let ez1 = z.mul_adde(e, Complex::ONE);
    let p = (ez1 + ez1).sqrt();
    let p2 = p.square();
    let p3 = p2 * p;

    // Both coefficients are real, so these are the componentwise (single-rounding) FMA.
    let odd = p3.mul_adde(V::splat(c[0]), p);
    let even = p2.nmul_adde(<V as FloatConsts>::FRAC_1_3, Complex::NEG_ONE);

    let w0_branch = even + odd;
    let wm1_branch = even - odd;

    // --- Logarithmic guess, W_k(z) ~ L1 - L2 + L2/L1, L1 = ln z + 2*pi*i*k ---
    let lnz = z.ln_p::<P>();
    let l1_0 = lnz;
    let l1_m1 = Complex::new(lnz.re, lnz.im - <V as FloatConsts>::TAU);

    let l2_0 = l1_0.ln_p::<P>();
    let l2_m1 = l1_m1.ln_p::<P>();

    let w0_log = l1_0 - l2_0 + l2_0 / l1_0;
    let wm1_log = l1_m1 - l2_m1 + l2_m1 / l1_m1;

    // --- Middle region for W_0 ---
    // ez/(2 + ez) is exact at z = -1/e and at z = 0 and decent between, covering the
    // gap where L1 = ln z passes through zero and the logarithmic guess degenerates.
    let ez = z * e;
    let w0_mid = ez / (ez + Complex::real(V::TWO));

    // --- Region selection ---
    let d = Complex::new(z.re + <V as FloatConsts>::FRAC_NEG_1_E.abs(), z.im);
    let near_branch = d.norm_sqr().cmp_lt(V::splat(c[1]));
    let far = z.norm_sqr().cmp_gt(e * e);

    let mut w0 = near_branch.select(w0_branch, far.select(w0_log, w0_mid));
    let mut wm1 = near_branch.select(wm1_branch, wm1_log);

    // --- Halley ---
    let mut n = 0;
    while n < iters {
        n += 1;
        w0 = halley::<P, V>(w0, z);
        wm1 = halley::<P, V>(wm1, z);
    }

    // --- Edge cases ---
    if const { P::POLICY.check_overflow } {
        let zero = z.re.is_zero() & z.im.is_zero();
        let nonzero = !zero;

        // W_0(0) = 0; W_{-1}(0) = -inf, the real convention for the limit.
        w0 = nonzero.select(w0, Complex::ZERO);
        wm1 = nonzero.select(wm1, Complex::new(V::NEG_INFINITY, V::ZERO));
    }

    (w0, wm1)
}

/// One Halley step for `w e^w = z`, written through `e^{-w}` so that neither tail
/// overflows:
///
/// ```text
/// g = w - z e^{-w}
/// d = (w^2 + 2w + 2) + (w + 2) z e^{-w}
/// w' = w - 2(w + 1) g / d
/// ```
#[inline(always)]
fn halley<P: Policy, V: RealFloatVector>(w: Complex<V>, z: Complex<V>) -> Complex<V> {
    let enw = (-w).exp_p::<P>();
    let wp1 = w + Complex::ONE;
    let zenw = z * enw;

    // w^2 + 2w + 2 = (w + 1)^2 + 1, and each product below folds its addend into the
    // complex FMA rather than rounding the product first.
    let q = wp1.mul_adde(wp1, Complex::ONE);

    let g = w - zenw;
    let d = (wp1 + Complex::ONE).mul_adde(zenw, q);

    (wp1 + wp1).nmul_adde(g / d, w)
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
    fn complex_lambert_w<P: Policy>(self) -> (Self, Self) {
        const C: [f32; 2] = [11.0 / 72.0, 0.09];
        lambert_w_impl::<P, V>(self, &C, 3)
    }

    #[inline(always)]
    fn faddeeva_w<P: Policy>(self) -> Self {
        self::faddeeva::faddeeva_w::<P, f32, f32, V>(self)
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
    fn complex_lambert_w<P: Policy>(self) -> (Self, Self) {
        const C: [f64; 2] = [11.0 / 72.0, 0.09];
        lambert_w_impl::<P, V>(self, &C, 4)
    }

    #[inline(always)]
    fn faddeeva_w<P: Policy>(self) -> Self {
        self::faddeeva::faddeeva_w::<P, f64, f64, V>(self)
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

decl_complex_math! {
    /// Complex special functions that carry an element-specific coefficient table.
    ///
    /// Generated from [`SpecializedComplexSpecialMath`] by the same macro that builds
    /// [`ComplexMath`](crate::math::ComplexMath), so `z.faddeeva_w_p::<Precision>()` behaves as
    /// `z.norm_p::<Precision>()` does and generic code can bound on `V: ComplexSpecialMath`.
    ///
    /// Only the members with no home in [`SpecialMath`](thermite_special::SpecialMath) are
    /// re-exposed here. The Gamma family is already `z.tgamma()` there; declaring it a
    /// second time would make every such call ambiguous whenever both traits are in scope.
    trait ComplexSpecial<FloatElement>: ComplexVector {
        /// `$w(z) = e^{-z^2}\operatorname{erfc}(-iz)$`, the Faddeeva function (also the
        /// complex error function, or the plasma dispersion function up to a factor).
        ///
        /// See [`faddeeva`] for the algorithm, the accuracy ladder, and the one
        /// caveat that matters (`$\operatorname{Re} w$` near the real axis).
        fn faddeeva_w[][](self: Self) -> Self;

        /// `$\operatorname{erfcx}(z) = e^{z^2}\operatorname{erfc}(z) = w(iz)$`, the
        /// scaled complementary error function.
        fn erfcx[][](self: Self) -> Self;

        /// The Voigt function `$K(x, y) = \operatorname{Re} w(x + iy)$`, as a real value.
        fn voigt[][](self: Self) -> Self::Real;
    }
}
