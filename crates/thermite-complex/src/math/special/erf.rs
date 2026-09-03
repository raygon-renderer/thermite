//! `erf` and `erfc` over the right half-plane: a Taylor series inside `|z| ~ 8.9` and the
//! Faddeeva function outside it, blended per lane. The reflections to the left half-plane
//! are in the `SpecializedSpecialMath` impl.

use thermite::math::TranscendentalMathWithPolicy as _;
use thermite::math::policy::{Policy, PrecisionPolicy};
use thermite::prelude::*;

use crate::Complex;
use crate::vector::RealFloatVector;

use super::SpecializedComplexSpecialMath;

/// Terms of the Taylor series, which needs roughly `2|z|^2` of them.
///
/// The loop exits once every lane has converged, after ~20 for typical arguments,
/// so this cap only costs the large-`|z|` lanes that need it.
const SERIES_TERMS: usize = 160;

/// `$|z|^2$` past which the series cannot converge within [`SERIES_TERMS`].
///
/// The series needs roughly `$2|z|^2$` terms, so 160 of them reach `$|z| \approx 8.9$`.
const SERIES_RADIUS_SQ: i64 = 64;

/// `$2\,\mathrm{Im}(z)^2$` past which the series cancels badly enough to be abandoned.
///
/// The series terms peak near `$e^{\lvert z\rvert^2}$` while the sum they build is
/// `$\tfrac{\sqrt\pi}{2}e^{z^2}\mathrm{erf}(z)$`, so the digits lost are
///
/// ```math
/// \log_{10}\frac{e^{\lvert z\rvert^2}}{\lvert e^{z^2}\rvert}
///   = \frac{\lvert z\rvert^2 - \mathrm{Re}(z^2)}{\ln 10}
///   = \frac{2\,\mathrm{Im}(z)^2}{\ln 10}
/// ```
///
/// `$\mathrm{Im}(z)$` **alone** governs it, not `$\mathrm{Re}(z^2)$`. Pulling
/// `$e^{-z^2}$` out front removes the alternation for real `z`, where the peak term and
/// the sum are the same size. Every unit of `$\mathrm{Im}(z)^2$` puts two back.
///
/// This limit was previously compared against `$-\mathrm{Re}(z^2) = y^2 - x^2$`, which
/// is the same quantity **only on the imaginary axis**, where it was tested. Along any
/// ray at 45 degrees `$\mathrm{Re}(z^2)$` is exactly zero however large `z` grows, so
/// the guard never fired and the series ran at a cancellation of `$e^{2y^2}$`: measured,
/// `erf` on `$z = a(1-i)$` was wrong by 2e-6 at `$\lvert z\rvert = 5.7$` and returned
/// `-3.0e10` at `$\lvert z\rvert = 8$`, where `$\lvert\mathrm{erf}\rvert \le 1.2$`.
/// The value 8 is unchanged and still means ~3.5 digits. Only the quantity it is
/// compared against is corrected.
///
/// Inside the remaining band the series is the better regime and is _structurally_
/// exact in ways `w` is not: for purely imaginary `z` every term is purely imaginary,
/// so `erf(iy)` has a real part of exactly zero.
///
/// # Why the limit is policy-dependent
///
/// The series' loss is `$e^{2y^2}$` **whatever the policy**: it is cancellation, and no
/// tier buys it back. `w`'s error is the opposite: measured 2.9e-13 at `Best` and 4.2e-10
/// at `Average`. So the tier decides which regime is better near the boundary. Setting
/// the series' loss equal to `w`'s:
///
/// | tier | `w` | break-even `$2y^2$` | limit used |
/// |---|---|---|---|
/// | `Best`+ | 2.9e-13 | 7.2 | 8 |
/// | below | 4.2e-10 | 14.5 | 16 |
///
/// 16 is also exactly what the old guard did on the imaginary axis (`$-Re(z^2) \ge 8$`
/// is `$y^2 \ge 8$`, i.e. `$2y^2 \ge 16$`), so the lower tiers keep the behaviour they
/// were tuned for, including the exact-zero real part out to `$y = 2.83$`.
const SERIES_ALTERNATION_LIMIT: i64 = 16;

/// [`SERIES_ALTERNATION_LIMIT`] at `Best` and above, where `w` is accurate enough to be
/// worth taking earlier.
const SERIES_ALTERNATION_LIMIT_BEST: i64 = 8;

/// `$Re(z^2) = x^2 - y^2$` past which `erfc` must be computed directly rather than as
/// `1 - erf`.
///
/// `Re(z^2)`, not `|z|`, governs `$e^{-z^2}$`, hence how small `erfc` is, hence how
/// badly `1 - erf` cancels. By `x = 6` there is nothing left: `erf(6)` has rounded to
/// exactly 1.0 while `erfc(6)` is 2e-17.
const DIRECT_ERFC_LIMIT: i64 = 6;

/// `(erf(z), erfc(z))` for `Re z >= 0`. The reflections in the impl below need no more.
///
/// Both regimes are evaluated on every lane and blended, as lanes of a vector need not
/// agree on which applies.
///
/// # Accuracy
///
/// ~1 ulp over the whole half-plane. `erfc` comes from [`faddeeva`] via
/// `$\operatorname{erfc}(z) = e^{-z^2}w(iz)$` for `$|z| \ge 1$` and from the Taylor
/// series inside that. The two regimes have no gap between them.
///
/// This replaced a continued fraction keyed on `Re(z^2) >= 6`, which left a wedge of
/// large `|Im z|` where neither regime was valid: the series truncated before converging
/// and the continued fraction was never selected. Measured against a 50-digit oracle,
/// `erfc(0.1 + 10i)` was wrong by 36 orders of magnitude and is now good to 1e-13.
#[inline(always)]
pub(crate) fn erf_erfc_positive<P: Policy, E, V>(z: Complex<V>) -> (Complex<V>, Complex<V>)
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
    //  2. `2 Im(z)^2` past the policy's limit, where the series cancels. See
    //     [`SERIES_ALTERNATION_LIMIT`] for why that is the right quantity, what this
    //     used to be compared against, and what that cost on the 45-degree rays.
    //  3. `|z| > 8`, where the series truncates before it converges.
    //
    // Cases 2 and 3 together are the wedge, and neither was covered before: the
    // continued fraction was selected on `Re(z^2) >= 6`, so everything with large
    // `|Im z|` fell through to a series that could not deliver it.
    //
    // Inside the remaining band the series is kept: `w` at `Average` is 4e-10 where the
    // series is at the last ulp, and near the origin that gap is all that matters.
    // Widening this trades accuracy in the bulk for nothing.
    let cancels = z2
        .re
        .cmp_ge(thermite::const_splat!(int <V::Element>: DIRECT_ERFC_LIMIT));
    // 2 Im(z)^2, formed directly rather than as `norm_sqr() - z2.re`: the two agree
    // in exact arithmetic, but that spelling subtracts two quantities of size `x^2` to
    // land on a threshold of order 10.
    let im_sq = z.im * z.im;
    let two_im_sq = im_sq + im_sq;
    let alternates = if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
        two_im_sq.cmp_ge(thermite::const_splat!(int <V::Element>: SERIES_ALTERNATION_LIMIT_BEST))
    } else {
        two_im_sq.cmp_ge(thermite::const_splat!(int <V::Element>: SERIES_ALTERNATION_LIMIT))
    };
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
    // The e^{-z^2} factor absorbs the alternation where `Re(z^2) > 0`. Case 2 above is
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
            // on the loop-carried chain: `term` then waits a division _and_ a multiply
            // per iteration. Scaling the loop-invariant `two_z2` instead leaves one
            // complex multiply between successive terms, with the reciprocal issuing
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
    // `w` (a single reciprocal and an N-term Horner) where the continued fraction it
    // replaced took 32 complex divisions.
    if const { P::POLICY.avoid_branching } || use_w.any() {
        w_erfc = exp_nz2 * SpecializedComplexSpecialMath::faddeeva_w::<P>(Complex::new(-z.im, z.re));
    }

    let erf = use_w.select(one - w_erfc, series_erf);
    let erfc = use_w.select(w_erfc, one - series_erf);

    (erf, erfc)
}
