use super::{Compensated, CompensatedFloatVector};

use thermite::math::{RealMathWithPolicy, TranscendentalMathWithPolicy};
use thermite::prelude::*;

use thermite_special::specialized::{SpecializedRealPrimalMath, SpecializedRealSpecialMath, SpecializedSpecialMath};
use thermite_special::{RealSpecialMathWithPolicy, SpecialMathWithPolicy};

use crate::specialized::special::SpecializedCompensatedSpecialMath;

// Compensated is a single-value real, so it belongs in the "primal" tier and gains the
// value-and-derivative (`_d`) activation forms (via the trait defaults).
impl<V: CompensatedFloatVector> SpecializedRealPrimalMath<Compensated<V::Element>> for Compensated<V>
where
    V: SpecialMathWithPolicy + RealSpecialMathWithPolicy + RealMathWithPolicy,
    V: SpecializedCompensatedSpecialMath<V::Element>,
{
}

impl<V: CompensatedFloatVector> Compensated<V>
where
    V: RealMathWithPolicy,
{
    #[inline(always)]
    fn erf_internal_p<P: Policy>(self) -> (Self, Self) {
        let x = self;
        let abs_x = x.abs();

        // Series below this, continued fraction above.
        //
        // 2 rather than 3, because the series computes *erf* and erfc comes out of it as
        // 1 - erf: the cancellation in that subtraction is what sets erfc's accuracy, and
        // it grows with erf. erf(2) = 0.9953 costs ~8 bits, erf(3) = 0.99998 costs ~16 -
        // and erfc at 2.751 measured 89 bits against the 104 it holds below 1. Handing
        // [2, 3) to the continued fraction, which computes erfc directly, brings that to
        // 101 and erf with it (102 -> 111, since erf = 1 - erfc barely cancels when erfc
        // is the small one). Nothing below 2 or above 3 changes.
        //
        // This matters beyond erf: `erfinv` refines against erfc, and needs it most
        // exactly where it was weakest, since large x corresponds to y near 1.
        let use_series: V::Mask = abs_x
            .value()
            .cmp_lt(V::splat(<V::Element as FloatElement>::ConstInt::<2>::VALUE));

        let use_only_series = use_series.all();
        let use_only_cf = use_series.none();

        // --- Init Series (erf) ---
        // erf(x) = 2/sqrt(pi) * (x - x^3/3 + x^5/10 ...)
        let x2 = -x.square(); // -x^2 for alternating series
        let mut sum_s = abs_x;
        let mut term_s = abs_x;

        // --- Init Continued Fraction (erfc) ---
        // Lentz's method vars
        //
        // `tiny` is the stand-in for a denominator that came out non-positive, so it only
        // has to be negligible against any real term - but it also gets *reciprocated* on
        // the very first step, and that is what constrains it here.
        //
        // `MIN_POSITIVE` cannot be used: 1/2.2e-308 is 4.5e307, and compensated
        // multiplication splits its operands with Dekker's 2^27+1 factor, which overflows
        // to infinity for anything past ~1.3e300. The next `f *= c * d` then produced NaN,
        // which is why erf and erfc returned NaN for every |x| >= 3 - the entire
        // continued-fraction tail, the only regime that reaches this code.
        //
        // sqrt(MIN_POSITIVE)/EPSILON leaves both the sentinel and its reciprocal far
        // inside the splitter's range while staying utterly negligible as a floor.
        let tiny = Self::new(V::MIN_POSITIVE.sqrt() / <V as FloatVector>::EPSILON);
        let mut f = tiny;
        let mut a = Self::ONE; // a_1 = 1
        let mut c = tiny;
        let mut d = Self::ZERO;
        let b = abs_x; // `b` in Lentz's method is |x|

        let shift = if P::POLICY.unroll_loops { 2 } else { 0 };
        let mut i = 1;
        let max_i = (P::POLICY.max_iterations >> shift) + 1;

        // the inner loops are expensive, so avoid unnecessary work

        #[rustfmt::skip]
        let () = match (use_only_series, use_only_cf) {
            (true, true) => {
                // both true -> all values are NaN or Inf
            },
            (true, false) => while i < max_i {
                let next_i = i + (1 << shift);
                let prev_s = sum_s;

                for k in i..next_i {
                    // --- Series Update ---
                    // term *= -x^2 * (2k-1) / (k * (2k+1))
                    let k_f = k as i64;
                    let k2_p1 = (2 * k + 1) as i64;
                    let k2_m1 = (2 * k - 1) as i64;

                    let num = FloatElement::from_int(k2_m1);
                    let den = FloatElement::from_int(k_f * k2_p1);

                    term_s *= x2 * Self::from_fraction(V::splat(num), V::splat(den));

                    sum_s.accumulate_unnormalized(term_s);
                }

                if prev_s.cmp_eq(sum_s).all() {
                    // println!("erf converged at i={}", next_i - 1);
                    break;
                }

                i = next_i;
            },
            (false, true) => while i < max_i {
                let next_i = i + (1 << shift);
                let prev_f = f;

                for k in i..next_i {
                    // --- CF Update ---
                    // Lentz coefficients: a_k = (k-1)/2
                    if k > 1 {
                        a = Self::splat(FloatElement::from_int((k - 1) as i64)) * Self::HALF;
                    }

                    // Lentz steps: D = b + a*D, C = b + a/C
                    d = a.mul_adde(d, b); // D = b + a*D

                    c = b + a / c.max(tiny); // if C<=0 -> tiny
                    d = d.max(tiny).reciprocal_p::<P>(); // if D<=0 -> tiny

                    f *= c * d;
                }

                if prev_f.cmp_eq(f).all() {
                    // println!("erfc converged at i={}", next_i - 1);
                    break;
                }

                i = next_i;
            },
            (false, false) => while i < max_i {
                let next_i = i + (1 << shift);
                let prev_s = sum_s;
                let prev_f = f;

                for k in i..next_i {
                    // --- Series Update ---
                    // term *= -x^2 * (2k-1) / (k * (2k+1))
                    let k_f = k as i64;
                    let k2_p1 = (2 * k + 1) as i64;
                    let k2_m1 = (2 * k - 1) as i64;

                    let num = FloatElement::from_int(k2_m1);
                    let den = FloatElement::from_int(k_f * k2_p1);

                    term_s *= x2 * Self::from_fraction(V::splat(num), V::splat(den));

                    let mut new_sum_s = sum_s;
                    new_sum_s.accumulate_unnormalized(term_s);

                    sum_s = use_series.select(new_sum_s, sum_s); // but avoid overflowing the sum

                    // --- CF Update ---
                    // Lentz coefficients: a_k = (k-1)/2
                    if k > 1 {
                        a = Self::splat(FloatElement::from_int((k - 1) as i64)) * Self::HALF;
                    }

                    // Lentz steps: D = b + a*D, C = b + a/C
                    d = a.mul_adde(d, b); // D = b + a*D

                    c = b + a / c.max(tiny); // if C<=0 -> tiny
                    d = d.max(tiny).reciprocal_p::<P>(); // if D<=0 -> tiny

                    f = use_series.select(f, f * c * d);
                }

                let series_converged = use_only_cf || prev_s.cmp_eq(sum_s).all();
                let cf_converged = use_only_series || prev_f.cmp_eq(f).all();

                if series_converged && cf_converged {
                    // println!("erf converged at i={}", next_i - 1);
                    break;
                }

                i = next_i;
            },
        };

        // --- Finalize ---

        // 1. Result from Series, also normalizes the compensated sum
        let res_erf_s = sum_s * Self::FRAC_2_SQRT_PI;

        // 2. Result from CF (if used)
        // erfc = e^(-x^2)/sqrt(pi) * f
        let res_erfc_c = if use_only_series {
            Self::ZERO // avoid doing exp if not needed
        } else {
            f * (-x.square()).exp_p::<P>() * Self::FRAC_1_SQRT_PI
        };

        // 3. Select based on Method
        // If series used: erf = res_erf_s,         erfc = 1 - res_erf_s
        // If CF used:     erf = 1 - res_erfc_c,    erfc = res_erfc_c

        let erf_val = use_series.select(res_erf_s, Self::ONE - res_erfc_c);
        let erfc_val = use_series.select(Self::ONE - res_erf_s, res_erfc_c);

        // 4. Symmetry for x < 0
        // erf(-x) = -erf(x)
        // erfc(-x) = 2 - erfc(x)

        let is_neg = x.value().is_negative();

        let final_erf = erf_val.neg_c(is_neg); // is_neg.select(-erf_val, erf_val);
        let final_erfc = is_neg.select(Self::TWO - erfc_val, erfc_val);

        (final_erf, final_erfc)
    }
}

impl<V: CompensatedFloatVector> SpecializedRealSpecialMath<Compensated<V::Element>> for Compensated<V>
where
    V: SpecialMathWithPolicy + RealSpecialMathWithPolicy + RealMathWithPolicy,
    V: SpecializedCompensatedSpecialMath<V::Element>,
{
    #[inline(always)]
    fn erfinv<P: Policy>(self) -> Self {
        // High-performance erfinv using Halley's Method seeded by Winitzki's approximation.
        // This converges in ~3 iterations for 106-bit precision.
        // However, for very small |y|, we can do better with the Maclaurin series expansion,
        // despite more iterations, since it avoids expensive calls to log/exp/sqrt functions.

        let y = self;
        let abs_y = y.abs();
        let y_value = y.value();
        let abs_y_value = abs_y.value();

        if abs_y_value.cmp_le(V::MAX_ERFINV_SERIES).all() {
            // For small |y|, use the Maclaurin series expansion for better performance,
            // since it doesn't need to call log/exp/sqrt/etc. functions.

            // Maclaurin series for erf_inv(y):
            // erf_inv(y) = sum_{k=0 to inf} (c_k / (2k+1)) * (sqrt(pi)/2 * y)^(2k+1)
            // where c_0 = 1, c_k = sum_{m=0 to k-1} (c_m * c_{k-1-m}) / ((m+1)(2m+1))

            let w = abs_y * Self::FRAC_SQRT_PI_2; // Variable w = (sqrt(pi)/2) * |y|
            let w2 = w.square();

            let mut sum = w; // Initial term (k=0): c_0 = 1, term = w
            let mut w_pow = w; // Stores w^(2k+1)

            const MAX_COEFFS: usize = 64;

            // Scalar Coefficient history buffer
            // We need this to compute the convolution for the next c_k.
            // 64 terms is generally sufficient for convergence where defined,
            // though it gets slow near |y| ~ 1.
            let mut coeffs: [Compensated<V::Element>; MAX_COEFFS] = [Element::ZERO; MAX_COEFFS];

            coeffs[0] = Element::ONE; // c_0 = 1

            let max_k = P::POLICY.max_iterations.min(MAX_COEFFS - 1);

            for k in 1..max_k {
                let prev = sum;

                let mut c_k: Compensated<V::Element> = Element::ZERO;

                for m in 0..k {
                    // Term: (c_m * c_{k-1-m}) / ((m+1)(2m+1))
                    let num = coeffs[m] * coeffs[k - 1 - m];

                    let m_i = m as i64;
                    let den_i = (m_i + 1) * (2 * m_i + 1);

                    c_k.accumulate_unnormalized(num / <V::Element as FloatElement>::from_int(den_i));
                }

                coeffs[k] = c_k.normalize();

                // Term = (c_k / (2k+1)) * w^(2k+1)
                w_pow *= w2; // Next odd power of w

                let k_term_den = (2 * k + 1) as i64;

                sum.accumulate_unnormalized(Self::splat(c_k) * w_pow / V::splat(FloatElement::from_int(k_term_den)));

                // Check for convergence
                if prev.cmp_eq(sum).all() {
                    // Restore sign: erf_inv(-y) = -erf_inv(y)
                    sum.value = sum.value.mul_sign(y_value);
                    sum.error = sum.error.mul_sign(y_value);

                    return sum.normalize();
                }
            }

            // The MAX_ERFINV_SERIES cutoff should guarantee convergence,
            // but if we reach here, we fallback to Halley's method.
            // This should be very rare, if not impossible.
        }

        // Detect singularities
        let is_zero = abs_y_value.cmp_eq(V::ZERO);
        let is_one = abs_y_value.cmp_eq(V::ONE);

        // 1. Initial guess: the inner vector's own `erfinv`, which is already correct to
        //    the element's full width (~53 bits for f64).
        //
        //    This used to be Winitzki's approximation, good to a relative 3.5e-4 - about
        //    11 bits. Halley is cubic, so 11 bits needs three passes to clear 106 and 53
        //    bits needs one, and every pass costs a compensated `erf` *and* a compensated
        //    `exp` to form f and f'. Seeding from the cheaper, far better starting point
        //    trades a scalar `erfinv` for two of each. The loop below is unchanged and
        //    still exits on its own convergence test, so a seed that ever disappoints
        //    simply iterates again rather than returning something wrong.
        let mut x = Self::new(abs_y_value.erfinv_p::<P>());

        // 2. Halley's Method Iterations (Cubic Convergence)
        // x_{n+1} = x_n - u / (1 + x_n * u) where u = f(x_n) / f'(x_n)
        let skip = is_zero | is_one; // cannot be solved as roots

        for _ in 0..P::POLICY.max_iterations {
            let prev_x = x;
            // f = erf(x) - y, routed through erf = 1 - erfc so that near y = 1 the two
            // quantities being subtracted are both *small* rather than both near 1.
            //
            // Measured, this changes nothing: erfinv(0.9999) sits at 3.43e-27 either way,
            // bit for bit. The cancellation it avoids is not the one that limits this -
            // f -> 0 at the root by definition, so some cancellation is unavoidable, and
            // the compensated erfc was already exact at these arguments. Kept because it
            // is the better-conditioned spelling and costs nothing (the kernel computes
            // erf and erfc together), but the y -> 1 shortfall has a different cause that
            // is not yet identified.
            let f = (Self::ONE - abs_y) - x.erfc_p::<P>();

            // f / f'(x) = f * (sqrt(pi)/2) * exp(x^2)
            let u = f * (Self::FRAC_SQRT_PI_2 * x.square().exp_p::<P>());

            // Halley step: u / (1 + x*u)
            // Note: f''/f' = -2x, so the Halley term simplifies to this.
            x.reduce_unnormalized(u / x.mul_adde(u, V::ONE));

            if (skip | x.cmp_eq(prev_x)).all() {
                break;
            }
        }

        x = is_zero.select(Self::ZERO, is_one.select(Self::INFINITY, x));

        // Restore sign: erf_inv(-y) = -erf_inv(y)
        x.value = x.value.mul_sign(y_value);
        x.error = x.error.mul_sign(y_value);

        x.normalize()
    }

    // Exact identity: probit(p) = sqrt(2) * erfinv(2p - 1), with every step in
    // compensated arithmetic (2p - 1 is an error-free transform here, and erfinv
    // refines to full double-double precision via Halley's method).
    #[inline(always)]
    fn probit<P: Policy>(self) -> Self {
        Self::erfinv::<P>(self + self - Self::ONE) * Self::SQRT_2
    }

    #[inline(always)]
    fn lgamma_r<P: Policy>(self) -> (Self, Self) {
        <V as SpecializedCompensatedSpecialMath<V::Element>>::compensated_lgamma_r::<P>(self)
    }
}

// The gamma family is blanket-implemented over `SpecializedCompensatedSpecialMath`, the
// same way `SpecialMath` is blanket-implemented over *this* trait. Adding a gamma method
// therefore touches only the lower rung; see `crate::specialized` for why that rung has
// to exist at all (the two Compensated widths need different coefficients).
impl<V: CompensatedFloatVector> SpecializedSpecialMath<Compensated<V::Element>> for Compensated<V>
where
    V: SpecialMathWithPolicy + RealSpecialMathWithPolicy + RealMathWithPolicy,
    V: SpecializedCompensatedSpecialMath<V::Element>,
{
    type ExpIntDetails = Self;

    #[inline(always)]
    fn erf<P: Policy>(self) -> Self {
        Self::erf_internal_p::<P>(self).0
    }

    #[inline(always)]
    fn erfc<P: Policy>(self) -> Self {
        Self::erf_internal_p::<P>(self).1
    }

    #[inline(always)]
    fn tgamma<P: Policy>(self) -> Self {
        <V as SpecializedCompensatedSpecialMath<V::Element>>::compensated_tgamma::<P>(self)
    }

    #[inline(always)]
    fn beta<P: Policy>(a: Self, b: Self) -> Self {
        <V as SpecializedCompensatedSpecialMath<V::Element>>::compensated_beta::<P>(a, b)
    }

    fn lambert_w<P: Policy>(self) -> (Self, Self) {
        // Seed from the standard-precision lambert_w on the value field,
        // then refine each branch with a single compensated Halley iteration.
        //
        // Halley's iteration for w*e^w = x:
        //   ew = exp(w), f = w*ew - x, wp1 = w + 1
        //   d = 2*wp1^2*ew - (w+2)*f
        //   w' = w - 2*wp1*f / d

        let x = self;
        let (w0_seed, wm1_seed) = x.value.lambert_w_p::<P>();

        let mut w0 = Self::new(w0_seed);
        let mut wm1 = Self::new(wm1_seed);

        // One compensated Halley step per branch
        #[inline(always)]
        fn halley_refine<P: Policy, W>(w: Compensated<W>, x: Compensated<W>) -> Compensated<W>
        where
            W: CompensatedFloatVector + RealMathWithPolicy,
        {
            let ew = w.exp_p::<P>();
            let f = w.mul_sube(ew, x);
            let wp1 = w + W::ONE;
            let wp2 = wp1 + wp1;
            let d = (wp1 + W::ONE).nmul_adde(f, wp2 * wp1 * ew);
            wp2.nmul_adde(f / d, w)
        }

        w0 = halley_refine::<P, V>(w0, x);
        wm1 = halley_refine::<P, V>(wm1, x);

        // Edge cases
        let x_val = x.value();
        let at_branch = x_val.cmp_eq(FloatConsts::FRAC_NEG_1_E);
        let at_zero = x_val.is_zero();

        w0 = at_branch.select(Self::NEG_ONE, w0);
        w0 = at_zero.select(Self::ZERO, w0);
        wm1 = at_branch.select(Self::NEG_ONE, wm1);
        wm1 = at_zero.select(Self::new(V::NEG_INFINITY), wm1);

        if const { P::POLICY.check_overflow } {
            let in_domain = x_val.cmp_ge(FloatConsts::FRAC_NEG_1_E);

            w0 = in_domain.select(w0, Self::NAN);
            w0 = x_val.cmp_eq(V::INFINITY).select(Self::INFINITY, w0);

            wm1 = in_domain.select(wm1, Self::NAN);
            wm1 = x_val.cmp_gt(V::ZERO).select(Self::NAN, wm1);
        }

        (w0, wm1)
    }

    fn lgamma<P: Policy>(self) -> Self {
        Self::lgamma_r::<P>(self).0
    }

    #[inline(always)]
    fn digamma<P: Policy>(self) -> Self {
        <V as SpecializedCompensatedSpecialMath<V::Element>>::compensated_digamma::<P>(self)
    }

    #[inline(always)]
    fn trigamma<P: Policy>(self) -> Self {
        <V as SpecializedCompensatedSpecialMath<V::Element>>::compensated_trigamma::<P>(self)
    }

    // TEMP(bessel_j): disabled until orders beyond J_0 exist - see thermite-special/src/lib.rs.
    //fn bessel_j<P: Policy, const N: usize>(self) -> Self {
    //    todo!()
    //}
}

/// Double-double is still real arithmetic, so the regime and domain rules apply
/// unchanged - but the Lentz sentinel does not.
impl<V: CompensatedFloatVector> thermite_special::specialized::ExpIntDetails<Compensated<V::Element>, Compensated<V>>
    for Compensated<V>
where
    Compensated<V>: thermite::vector::FloatVector<Element = Compensated<V::Element>>,
{
    /// The default, `MIN_POSITIVE`, is reciprocated on the first Lentz step, and
    /// 1/2.2e-308 = 4.5e307 is past the ~1.3e300 where compensated multiplication's
    /// Dekker 2^27+1 splitter overflows to infinity - so every continued-fraction lane
    /// came back NaN. `expint` takes the fraction for x >= 1, which is exactly where it
    /// failed.
    ///
    /// Same defect and same fix as `Complex`, and as the `erf`/`erfc` tail in this crate:
    /// a sentinel only has to be negligible as a *floor*, but this one also has to
    /// survive being inverted.
    #[inline(always)]
    fn cf_tiny() -> Compensated<V> {
        Compensated::new(V::MIN_POSITIVE.sqrt() / <V as FloatVector>::EPSILON)
    }
}
