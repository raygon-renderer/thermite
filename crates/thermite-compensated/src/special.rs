use super::{Compensated, CompensatedFloatVector};

use thermite::math::TranscendentalMathWithPolicy;
use thermite::prelude::*;

use thermite_special::SpecialMathWithPolicy;
use thermite_special::specialized::{SpecializedRealPrimalMath, SpecializedRealSpecialMath, SpecializedSpecialMath};

// Compensated is a single-value real, so it belongs in the "primal" tier and gains the
// value-and-derivative (`_d`) activation forms (via the trait defaults).
impl<V: CompensatedFloatVector> SpecializedRealPrimalMath<Compensated<V::Element>> for Compensated<V> where
    V: SpecialMathWithPolicy
{
}

impl<V: CompensatedFloatVector> Compensated<V>
where
    V: TranscendentalMathWithPolicy,
{
    #[inline(always)]
    fn erf_internal_p<P: Policy>(self) -> (Self, Self) {
        let x = self;
        let abs_x = x.abs();

        // threshold can be tuned
        let use_series: V::Mask = abs_x
            .value()
            .cmp_lt(V::splat(<V::Element as FloatElement>::ConstInt::<3>::VALUE));

        let use_only_series = use_series.all();
        let use_only_cf = use_series.none();

        // --- Init Series (erf) ---
        // erf(x) = 2/sqrt(pi) * (x - x^3/3 + x^5/10 ...)
        let x2 = -x.square(); // -x^2 for alternating series
        let mut sum_s = abs_x;
        let mut term_s = abs_x;

        // --- Init Continued Fraction (erfc) ---
        // Lentz's method vars
        let tiny = Self::MIN_POSITIVE;
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
    V: SpecialMathWithPolicy,
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

        // 1. Initial Guess via Winitzki's Approximation, using non-compensated math
        //    since the initial guess does not need to be incredibly accurate.
        // Relative error < 0.00035 across the domain.
        // Original: x ~ sqrt( sqrt(T1^2 - T2) - T1 )
        // Stable:   x ~ sqrt( -T2 / (sqrt(T1^2 - T2) + T1) )
        // This avoids catastrophic cancellation when y -> 0 (and thus T2 -> 0).

        // Constants
        let a = V::splat(FloatElement::from_ratio(147, 1000)); // a = 0.147
        let c = <V as FloatConsts>::FRAC_2_PI / a; // C = 2 / (pi * a)
        // L = ln(1 - y^2), use ln_1p for accuracy: ln(1 - y^2) = ln_1p(-y^2)
        let l = (-y.square().value()).ln_1p_p::<P>();

        let half_l = l * V::HALF;
        let t1 = c + half_l;
        let t2 = l / a;

        // Stable Winitzki guess
        let root_term = t1.mul_sube(t1, t2).sqrt();
        let inner = -t2 / (root_term + t1);

        // Clamp inner to 0 to avoid NaN if y ~ 0 results in tiny negative due to noise
        let mut x = Self::new(inner.max(V::ZERO).sqrt());

        // 2. Halley's Method Iterations (Cubic Convergence)
        // x_{n+1} = x_n - u / (1 + x_n * u) where u = f(x_n) / f'(x_n)
        let skip = is_zero | is_one; // cannot be solved as roots

        for i in 0..P::POLICY.max_iterations {
            let prev_x = x;
            let f = x.erf_p::<P>() - abs_y; // Work with absolute y for stability

            // f / f'(x) = f * (sqrt(pi)/2) * exp(x^2)
            let u = f * (Self::FRAC_SQRT_PI_2 * x.square().exp_p::<P>());

            // Halley step: u / (1 + x*u)
            // Note: f''/f' = -2x, so the Halley term simplifies to this.
            x.reduce_unnormalized(u / x.mul_adde(u, V::ONE));

            if (skip | x.cmp_eq(prev_x)).all() {
                // println!("erf_inv converged in Halley in {} iterations", i);
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

    fn lgamma_r<P: Policy>(self) -> (Self, Self) {
        todo!()
    }
}

impl<V: CompensatedFloatVector> SpecializedSpecialMath<Compensated<V::Element>> for Compensated<V>
where
    V: SpecialMathWithPolicy,
{
    #[inline(always)]
    fn erf<P: Policy>(self) -> Self {
        Self::erf_internal_p::<P>(self).0
    }

    #[inline(always)]
    fn erfc<P: Policy>(self) -> Self {
        Self::erf_internal_p::<P>(self).1
    }

    fn tgamma<P: Policy>(self) -> Self {
        todo!()
    }

    fn beta<P: Policy>(a: Self, b: Self) -> Self {
        todo!()
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
            W: CompensatedFloatVector + TranscendentalMathWithPolicy,
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

    fn digamma<P: Policy>(self) -> Self {
        todo!()
    }

    fn bessel_j<P: Policy, const N: usize>(self) -> Self {
        todo!()
    }
}
