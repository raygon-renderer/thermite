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
//! # Not implemented
//!
//! The Gamma family (`tgamma`, `lgamma`, `digamma`, `beta`) is a Lanczos
//! approximation whose coefficient tables are private to `thermite-special`'s
//! per-element impls. Lanczos itself extends to C (with the reflection formula for
//! `Re z < 1/2`), so exposing those tables is the prerequisite.
//!
//! `lambert_w` returns the real branches `(W_0, W_-1)`, a signature that does not
//! carry over to the countable family `W_k` over C. `bessel_j` only implements
//! `J_0` upstream. `expint` picks its regime by comparing `|x|` against 1, an
//! ordering C does not have.
//!
//! These `todo!()`; everything not depending on them works.

use thermite::math::policy::Policy;
use thermite::prelude::*;
use thermite_special::specialized::SpecializedSpecialMath;

use crate::Complex;
use crate::vector::ComplexFloatVector;

/// Terms of the Taylor series, which needs roughly `2|z|^2` of them.
///
/// The loop exits once every lane has converged, after ~20 for typical arguments,
/// so this cap only costs the large-`|z|` lanes that need it.
const SERIES_TERMS: usize = 160;

/// Iterations of the continued fraction. Worst-case 8e-15 relative over the region
/// it is used in, against a 50-digit oracle.
const CF_TERMS: usize = 32;

/// `$Re(z^2) = x^2 - y^2$` past which the continued fraction is used.
///
/// `Re(z^2)`, not `|z|`, governs `$e^{-z^2}$`, hence how small `erfc` is, hence how
/// badly `1 - erf` cancels. Past this line `erf` is close enough to 1 that the
/// subtraction loses digits: by `x = 6` it has none left, `erf(6)` having rounded
/// to exactly 1.0 while `erfc(6)` is 2e-17. So `erfc` is taken from the continued
/// fraction there and `erf` becomes the subtraction, which does not cancel.
///
/// 6 is where the two error curves cross. Above it the continued fraction is good
/// to 8e-15; below it `erfc` is no smaller than 5e-4, so `1 - erf` still holds ~13
/// digits.
const CF_LIMIT: i64 = 6;

impl<V: ComplexFloatVector> Complex<V> {
    /// `(erf(z), erfc(z))` for `Re z >= 0`. The reflections below need no more.
    ///
    /// Both regimes are evaluated on every lane and blended, the lanes of a vector
    /// not agreeing on which applies.
    ///
    /// # Accuracy
    ///
    /// ~1 ulp over `Re(z^2) >= 6` for any `|z|`, and over `|z| <~ 8` elsewhere. The
    /// gap is the wedge of large `|Im z|` with `|z| > 8`, where [`SERIES_TERMS`] cuts
    /// the Taylor series off before it converges, and where `erf` is of order
    /// `$e^{|Im z|^2}$` and has already overflowed f32. Closing it means a Faddeeva
    /// `w(z)`.
    #[inline(always)]
    fn erf_erfc_positive<P: Policy>(self) -> (Self, Self) {
        let z = self;
        let z2 = z.square();
        let exp_nz2 = (-z2).exp_p::<P>(); // e^{-z^2}, common to both regimes

        // --- Taylor series (Abramowitz & Stegun 7.1.6) ---
        //
        //   erf(z) = (2/sqrt(pi)) e^{-z^2} * sum_{n>=0} t_n,
        //     t_0 = z,  t_n = t_{n-1} * 2z^2 / (2n + 1)
        //
        // The e^{-z^2} factor absorbs the alternation. Every term then carries the
        // sign of z, and none of the cancellation the naive series suffers past |z| = 3.
        let two_z2 = z2 + z2;

        let mut term = z;
        let mut sum = z;

        let eps: V = <V as FloatVector>::EPSILON;

        let mut n = 1usize;
        while n < SERIES_TERMS {
            let denom = V::splat(<V::Element as FloatElement>::from_int(2 * n as thermite::LargeInt + 1));

            term = term * two_z2 / denom;
            sum += term;

            // |term| <= eps * |sum| in every lane: no lane can still change.
            let converged = term.norm_sqr().cmp_le(sum.norm_sqr() * (eps * eps));

            if converged.all() {
                break;
            }

            n += 1;
        }

        let series_erf = sum * exp_nz2 * <V as thermite::math::FloatConsts>::FRAC_2_SQRT_PI;

        // --- Continued fraction, for the erfc side ---
        //
        //   erfc(z) = (e^{-z^2}/sqrt(pi)) * 1/(z + (1/2)/(z + 1/(z + (3/2)/(z + ...))))
        //
        // with a_k = (k-1)/2. The asymptotic expansion is the usual alternative and is
        // not good enough: its error bottoms out at ~e^{-x^2} relative however it is
        // truncated (1e-11 at x = 5, 8e-8 at x = 4), leaving a band around x ~ 4 where
        // neither it nor the series reaches f64 accuracy. The continued fraction has
        // no such gap.
        //
        // Evaluated bottom-up from the truncation at CF_TERMS, not by a forward
        // (modified Lentz) sweep. Lentz needs a tiny non-zero seed, and a tiny complex
        // seed is fatal: the reciprocal goes through conj(w)/|w|^2, and |w|^2 of a
        // near-underflow value flushes to zero, giving an infinity and then a NaN.
        // Backward evaluation needs no seed, and takes one complex division per term
        // where Lentz takes two.
        //
        // No zero-guards: this branch is only selected where Re(z^2) >= 6, which
        // forces Re z > 0, and with every a_k > 0 the denominators stay in the right
        // half-plane. Lanes outside that region may divide by zero here; the select
        // below discards them.
        let mut w = Self::ZERO;

        let mut k = CF_TERMS;
        while k >= 2 {
            let a: V = V::splat(<V::Element as FloatElement>::from_ratio(k as thermite::LargeInt - 1, 2));

            // w = a_k / (z + w)
            w = (z + w).reciprocal_p::<P>() * a;

            k -= 1;
        }

        let cf_erfc = (z + w).reciprocal_p::<P>() * exp_nz2 * <V as thermite::math::FloatConsts>::FRAC_1_SQRT_PI;

        // --- blend ---
        //
        // Neither regime is valid where the other applies, and the select discards the
        // wrong one lane-by-lane. One mask serves both: past the cancellation line erfc
        // is the computed one and erf the subtraction, and below it the reverse.
        let limit = V::splat(<V::Element as FloatElement>::from_int(CF_LIMIT));
        let use_cf = z2.re.cmp_ge(limit);

        let erf = use_cf.select(Self::ONE - cf_erfc, series_erf);
        let erfc = use_cf.select(cf_erfc, Self::ONE - series_erf);

        (erf, erfc)
    }
}

impl<V: ComplexFloatVector> SpecializedSpecialMath<Complex<V::Element>> for Complex<V> {
    /// The error function over the whole complex plane.
    ///
    /// `erf` is entire and odd. The negative-real half-plane comes from
    /// `erf(-z) = -erf(z)`, a conditional negation, not a branch.
    #[inline(always)]
    fn erf<P: Policy>(self) -> Self {
        let neg = self.re.is_negative();
        let z = Complex::new(self.re.neg_c(neg), self.im.neg_c(neg));

        let (erf, _) = z.erf_erfc_positive::<P>();

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

        let (_, erfc) = z.erf_erfc_positive::<P>();

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

    // The Gamma family: Lanczos, whose coefficient tables are private to
    // thermite-special's per-element impls. See the module docs.

    #[inline(always)]
    fn tgamma<P: Policy>(self) -> Self {
        todo!(
            "Complex tgamma needs thermite-special's Lanczos coefficients, which are private to its per-element impls"
        )
    }

    #[inline(always)]
    fn lgamma<P: Policy>(self) -> Self {
        todo!(
            "Complex lgamma needs thermite-special's Lanczos coefficients, which are private to its per-element impls"
        )
    }

    #[inline(always)]
    fn digamma<P: Policy>(self) -> Self {
        todo!("Complex digamma needs thermite-special's asymptotic/reflection coefficients, which are private to it")
    }

    #[inline(always)]
    fn beta<P: Policy>(_a: Self, _b: Self) -> Self {
        todo!("Complex beta is Gamma(a)Gamma(b)/Gamma(a+b); blocked on complex tgamma")
    }

    /// Over C the Lambert W function has a countable family of branches `W_k`, so
    /// the real `(W_0, W_-1)` signature does not carry over.
    #[inline(always)]
    fn lambert_w<P: Policy>(self) -> (Self, Self) {
        todo!(
            "Complex lambert_w needs a branch-index parameter; the real (W_0, W_-1) signature does not carry over to C"
        )
    }

    /// `thermite-special` only implements `J_0` (f32).
    #[inline(always)]
    fn bessel_j<P: Policy, const N: usize>(self) -> Self {
        todo!("Complex bessel_j is blocked on thermite-special, which only implements J_0")
    }

    /// The default picks between a power series and a continued fraction by comparing
    /// `|x|` against 1, an ordering C does not have.
    #[inline(always)]
    fn expint<P: Policy, const N: usize>(self) -> Self {
        todo!(
            "Complex expint needs a complex-specific series/continued-fraction split; the default regime test is an ordering on |x|"
        )
    }
}
