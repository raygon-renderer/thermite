#![allow(unused)]

use num_traits::{MulAdd, Signed};

use crate::{CompensatedElement, consts::CompensatedLogTable};

use super::{Compensated, CompensatedRegister};

use thermite::{
    Mask, Vector,
    math::{
        FloatConsts, MathWithPolicy,
        policy::{Policy, PrecisionPolicy},
    },
    register::{Element, FloatElement, FloatRegister, Register},
};

impl<R: CompensatedRegister> MathWithPolicy<R> for Compensated<R>
where
    Vector<R>: MathWithPolicy<R>,
    R::Element: CompensatedRegister<Element = R::Element>,
{
    #[inline(always)]
    fn ldexp_p<P: Policy>(self, exp: Vector<<R as FloatRegister>::Signed>) -> Self {
        let Compensated { mut value, mut error } = self;

        value = MathWithPolicy::ldexp_p::<P>(value, exp);
        error = MathWithPolicy::ldexp_p::<P>(error, exp);

        Self { value, error }
    }

    #[inline(always)]
    fn frexp_p<P: Policy>(self) -> (Self, Vector<<R as FloatRegister>::Signed>) {
        let (f, k) = self.value().frexp_p::<P>();
        (Self::from_parts(f, self.error.ldexp_p::<P>(-k)), k)
    }

    #[inline(always)]
    fn to_degrees_p<P: Policy>(self) -> Self {
        self * Self::FRAC_180_PI
    }

    #[inline(always)]
    fn to_radians_p<P: Policy>(self) -> Self {
        self * Self::FRAC_PI_180
    }

    #[inline(always)]
    fn tolerance_p<P: Policy>() -> Self {
        Self::EPSILON * Self::splat(FloatElement::from_i64(P::POLICY.precision.tolerance()))
    }

    #[inline(always)]
    fn poly_p<P: Policy, const N: usize>(self, coeffs: &[R::Element; N]) -> Self {
        if const { !P::POLICY.unroll_loops || P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            // basic Horner's method that's both compact and accurate, even without FMA
            let mut res = Self::splat(coeffs[N - 1]);
            for &c in coeffs.iter().rev().skip(1) {
                res = res.mul_add(self, Vector::splat(c));
            }
            return res;
        }

        fast_polynomial::poly_f_n::<_, _, N>(self, |i| unsafe { Self::splat(*coeffs.get_unchecked(i)) })
    }

    #[inline(always)]
    fn poly_rev_p<P: Policy, const N: usize>(self, coeffs: &[R::Element; N]) -> Self {
        if const { !P::POLICY.unroll_loops || P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            // basic Horner's method that's both compact and accurate, even without FMA
            let mut res = Self::splat(coeffs[0]);
            for &c in coeffs.iter().skip(1) {
                res = res.mul_add(self, Vector::splat(c));
            }
            return res;
        }

        fast_polynomial::poly_f_n::<_, _, N>(self, |i| unsafe { Self::splat(*coeffs.get_unchecked(N - 1 - i)) })
    }

    #[inline(always)]
    fn poly_rational_p<P: Policy, const N: usize, const D: usize>(
        self,
        numerator: &[R::Element; N],
        denominator: &[R::Element; D],
    ) -> Self {
        self.poly_p::<P, N>(numerator) / self.poly_p::<P, D>(denominator)
    }

    fn sum_f_p<P: Policy, F>(start: i64, end: i64, f: F) -> Result<Self, Self>
    where
        F: FnMut(i64) -> Self,
    {
        todo!()
    }

    fn prod_f_p<P: Policy, F>(start: i64, end: i64, f: F) -> Result<Self, Self>
    where
        F: FnMut(i64) -> Self,
    {
        todo!()
    }

    #[inline(always)]
    fn newtons_method_p<P: Policy, F>(self, tolerance: Self, bounds: Option<(Self, Self)>, mut f: F) -> Self
    where
        F: FnMut(Self) -> (Self, Self),
    {
        let mut x = self;
        let tolerance = tolerance.value();

        for i in 0..P::POLICY.max_iterations {
            let (y, y_prime) = f(x);
            let delta = y / y_prime;

            let mut stop = delta.value().abs().cmp_le(tolerance);

            if P::POLICY.check_overflow {
                stop |= y_prime.value().abs().cmp_le(tolerance);
            }

            if stop.all() {
                // println!("Converged in {} iterations", i);
                break;
            }

            x = stop.select(x, x - delta);

            if let Some((min, max)) = bounds {
                x = x.clamp(min, max);
            }
        }

        x
    }

    #[inline(always)]
    fn smoothstep_p<P: Policy, const N: usize>(self, edges: Option<(Self, Self)>) -> Self {
        use thermite::math::internal::Smoothstep;

        let mut t = self;

        if let Some((a, b)) = edges {
            t = (t - a) / (b - a);
        }

        if P::POLICY.check_overflow {
            t = t.clamp(Self::ZERO, Self::ONE);
        }

        match N {
            0 => Self::new(t.value().step_p::<P>(Vector::HALF)),
            1 => t,
            _ => {
                t.powi_p::<P>(N as i32)
                    * const { Smoothstep::<R::Element, N>::COEFFICIENTS }
                        .into_iter()
                        .fold(Self::ZERO, |res, c| {
                            res.mul_add(t, Vector::splat(FloatElement::from_i64(c)))
                        })
            }
        }
    }

    fn inverse_smoothstep_p<P: Policy, const N: usize>(self, edges: Option<(Self, Self)>) -> Self {
        todo!()
    }

    fn smoothstep_derivative_p<P: Policy, const N: usize>(self, edges: Option<(Self, Self)>) -> Self {
        todo!()
    }

    fn smooth_interpolator_p<P: Policy>(self, edges: Option<(Self, Self)>, k: Self) -> Self {
        todo!()
        // let mut t = self;

        // if let Some((a, b)) = edges {
        //     // rescale t to [0, 1]
        //     t = (t - a) / (b - a);
        // }

        // let kt = k * t;

        // // (2x-1) / (kx^2-kx)
        // let e = t.mul_sube(Self::TWO, Self::ONE) / kt.mul_sube(t, kt);

        // // exp(e) + 1
        // let d = e.exp_p::<P>() + Vf::ONE;

        // // 1/(exp(e) + 1), it's important this is done in extra precision
        // let mut res = d.reciprocal_p::<ExtraPrecision<P>>();

        // let overflow = e.is_infinite();

        // // If the denominator is small enough, it could cause overflow,
        // // however that only really happens when t is very close to 0 or 1,
        // // or when k is very small. So approximate it with a step function.
        // if P::POLICY.avoid_branching || crate::unlikely(overflow.any()) {
        //     res = overflow.select(t.step_p::<P>(Vf::HALF), res);
        // }

        // // these are important since the Exp formulation is discontinuous at 0 and 1,
        // // and this maintains the asymptotes when t is outside the range [0, 1]
        // res = t.cmp_ge(Self::ONE).select(Self::ONE, res);
        // res = t.cmp_le(Self::ZERO).select(Self::ZERO, res);

        // res
    }

    fn smooth_interpolator_inverse_p<P: Policy>(self, edges: Option<(Self, Self)>, k: Self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn step_p<P: Policy>(self, edge: Self) -> Self {
        // removes error component
        Self::new(self.value().step_p::<P>(edge.value()))
    }

    #[inline(always)]
    fn lerp_p<P: Policy>(self, a: Self, b: Self) -> Self {
        self.mul_add(b - a, a)
    }

    #[inline(always)]
    fn scale_p<P: Policy>(self, in_min: Self, in_max: Self, out_min: Self, out_max: Self) -> Self {
        let in_range = in_max - in_min;
        let t = (self - in_min) / in_range;
        t.mul_add(out_max - out_min, out_min)
    }

    #[inline(always)]
    fn reciprocal_p<P: Policy>(self) -> Self {
        let q = self.value.reciprocal_p::<P>();

        let (p, e_prod) = crate::two_prod(q, self.value);
        let e1 = Vector::ONE - p;

        let err = if const { !R::HAS_APPROX_RCP || P::POLICY.precision.ge(PrecisionPolicy::Average) } {
            // assume q is accurate enough that we can ignore p_lo and avoid the extra division
            q * self.error.nmul_adde(q, e1) // r / self.value
        } else {
            // calculate the remainder r and include it in the error term,
            // unfortunately we must use a real division here if reciprocal is approximate,
            // which kind of defeats the purpose of using an approximate reciprocal in the first place,
            // but at least we get a better error term.
            self.error.nmul_adde(q, e1 - e_prod) / self.value
        };

        Self::renormalized(q, err)
    }

    #[inline(always)]
    fn inverse_sqrt_p<P: Policy>(self) -> Self {
        // We don't use approximate rsqrt here,
        // since the whole point is to get a precise result.
        self.sqrt().reciprocal_p::<P>()
    }

    #[inline(always)]
    fn powi_p<P: Policy>(self, mut e: i32) -> Self {
        let mut x = self;
        let mut res = Self::ONE;

        if e < 0 {
            e = -e;
            x = Self::reciprocal_p::<P>(x);
        }

        while e != 0 {
            if e & 1 != 0 {
                res *= x;
            }

            x *= x;
            e >>= 1;
        }

        res
    }

    #[inline(always)]
    fn powiv_p<P: Policy>(self, mut e: Vector<R::Signed>) -> Self {
        let mut x = self;
        let mut res = Self::ONE;

        x = e.is_negative().select(x.reciprocal_p::<P>(), x);
        e = e.abs();

        loop {
            let mut e1 = e & Vector::ONE;

            let nx = res * x;

            // NOTE: e1 is bitcast to Self when `select` is used, so we use it for the MSB_BLENDV hack
            // requirements
            res = if <R as Register>::HAS_MSB_BLENDV {
                // Move the lowest bit to the highest bit position
                e1 <<= const { core::mem::size_of::<<R::Signed as Register>::Element>() as u32 * 8 - 1 };

                // Blend the result based on the highest bit of e1
                Mask::from_unchecked(e1).select(nx, res)
            } else {
                e1.cmp_ne(Vector::ZERO).select(nx, res)
            };

            x *= x;
            e >>= 1;

            if e.cmp_ne(Vector::ZERO).none() {
                return res;
            }
        }

        res
    }

    #[inline(always)]
    fn hypot_p<P: Policy>(self, y: Self) -> Self {
        if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
            self.mul_add(self, y * y).sqrt()
        } else {
            let x = self.abs();
            let y = y.abs();

            let (min, max) = x.min_max(y);

            let t = min / max;

            let mut res = max * t.mul_add(t, Vector::ONE).sqrt();

            if P::POLICY.check_overflow {
                let inf = Vector::INFINITY;
                res = (x.value().cmp_lt(inf) & y.value().cmp_lt(inf) & t.value().cmp_lt(inf)).select(res, x + y);
            }

            res
        }
    }

    #[inline(always)]
    fn sin_cos_p<P: Policy>(self) -> (Self, Self) {
        // 1. Argument Reduction
        // Reduce x to r in [-pi/4, pi/4]
        // k = round(x / (pi/2))

        // 1a. Calculate k = round(x / (pi/2)) = round(x * (2/pi))
        let k = (self * Self::FRAC_2_PI).value().round();
        let k_comp = Self::new(k);

        // 1b. Compute r = x - k * (pi/2)
        // We must use the Compensated constant FRAC_PI_2 for high precision subtraction.
        // x - (k * PI/2)
        let r = k_comp.mul_add(-Self::FRAC_PI_2, self);

        let r2 = -(r * r); // -r^2, used for iterative multiplication

        // 2. Series Expansion
        // sin(r) = r - r^3/3! + r^5/5! ...
        // cos(r) = 1 - r^2/2! + r^4/4! ...

        // Initialize sums and terms
        // Term indices:
        // k=1: sin term needs /2*3, cos term needs /1*2

        let mut sin = r;
        let mut term_s = r;

        let mut cos = Self::ONE;
        let mut term_c = Self::ONE;

        let shift = if P::POLICY.unroll_loops { 1 } else { 0 }; // Conservative unroll
        let mut i = 1;

        // Approx 20 iterations sufficient for full compensated precision
        let max_i = (P::POLICY.max_iterations >> shift) + 1;

        while i < max_i {
            let next_i = i + (1 << shift);
            let prev_s = sin;
            let prev_c = cos;

            for k in i..next_i {
                let k2 = (2 * k) as i64;

                // Update Cosine Term: prev_term * (-r^2) / ((2k-1)*2k)
                let div_c = (k2 - 1) * k2;
                term_c *= r2 / Vector::splat(FloatElement::from_i64(div_c));
                cos.accumulate_unnormalized(term_c);

                // Update Sine Term: prev_term * (-r^2) / (2k*(2k+1))
                let div_s = k2 * (k2 + 1);
                term_s *= r2 / Vector::splat(FloatElement::from_i64(div_s));
                sin.accumulate_unnormalized(term_s);
            }

            if (prev_s.cmp_eq(sin) & prev_c.cmp_eq(cos)).all() {
                // println!("trig converged at i={}", next_i - 1);
                break;
            }

            i = next_i;
        }

        // 3. Reconstruction
        // Determine the final sin/cos based on the quadrant k.
        // The quadrant mapping for sin(x) / cos(x) where x = k * pi/2 + r:
        // k % 4 == 0:  sin ->  s,   cos ->  c
        // k % 4 == 1:  sin ->  c,   cos -> -s
        // k % 4 == 2:  sin -> -s,   cos -> -c
        // k % 4 == 3:  sin -> -c,   cos ->  s

        // Use integer mask logic on k.
        // We can check the low 2 bits of k.
        let k_int: Vector<R::Signed> = k.cast();

        // Construct bitmasks
        let bit0 = (k_int & Vector::ONE).cmp_ne(Vector::ZERO); // true if k % 2 != 0 (quadrants 1, 3)
        let bit1 = (k_int & Vector::TWO).cmp_ne(Vector::ZERO); // true if k % 4 >= 2 (quadrants 2, 3)

        // Swap sin/cos if k is odd (quadrants 1, 3), and normalize the results
        let mut final_sin = bit0.select(cos, sin).normalize();
        let mut final_cos = bit0.select(sin, cos).normalize(); // Note: sign is handled next

        // Sign logic:
        // Sin sign: positive in 0, 1. Negative in 2, 3. -> invert if bit1 is true.
        // Cos sign: positive in 0, 3. Negative in 1, 2. -> invert if (bit0 ^ bit1) is true.

        let neg_sin = bit1;
        let neg_cos = bit0 ^ bit1;

        // prepare for conditional negate
        let neg_sin = neg_sin.cast();
        let neg_cos = neg_cos.cast();

        final_sin.value = final_sin.value.conditional_negate(neg_sin);
        final_sin.error = final_sin.error.conditional_negate(neg_sin);

        final_cos.value = final_cos.value.conditional_negate(neg_cos);
        final_cos.error = final_cos.error.conditional_negate(neg_cos);

        // Zero check: if input was zero, result should be exact zero/one
        // (This is implicitly handled by series, but overflow checks might be needed for large inputs)

        (final_sin, final_cos)
    }

    #[inline(always)]
    fn sin_p<P: Policy>(self) -> Self {
        self.sin_cos_p::<P>().0
    }

    #[inline(always)]
    fn cos_p<P: Policy>(self) -> Self {
        self.sin_cos_p::<P>().1
    }

    #[inline(always)]
    fn tan_p<P: Policy>(self) -> Self {
        self.sin_p::<P>() / self.cos_p::<P>()
    }

    #[inline(always)]
    fn sinc_p<P: Policy>(self) -> Self {
        // Use non-compensated 4th root epsilon for tiny check, since
        // the Taylor series is actually very good for very small x.
        let is_tiny = self.value().abs().cmp_lt(Vector::FOURTH_ROOT_EPSILON);

        let x2 = self * self;

        // if branching, use Taylor series for tiny x without calling sine.
        if !P::POLICY.avoid_branching && is_tiny.all() {
            let res = x2 / Vector::splat(FloatElement::from_i64(120));
            return x2.mul_add(res - Self::FRAC_1_6, Self::ONE);
        }

        // For very small x, sinc(x) ~ 1 - x^2/6 + x^4/120
        let num = is_tiny.select(x2, self.sin_p::<P>());
        let den = is_tiny.select(Self::splat(FloatElement::from_i64(120)), self);

        // combined division, since division is expensive
        let mut y = num / den;

        y = is_tiny.select(x2.mul_add(y - Self::FRAC_1_6, Self::ONE), y);

        if P::POLICY.check_overflow {
            y = self.value().is_infinite().select(Self::ZERO, y);
        }

        y
    }

    #[inline(always)]
    fn sin_pix_p<P: Policy>(self) -> Self {
        self * (self * Self::PI).sin_p::<P>()
    }

    #[inline(always)]
    fn sinh_p<P: Policy>(self) -> Self {
        // (e^x - e^-x) / 2
        // For small x, precision loss occurs with explicit subtract.
        // Use expm1: (expm1(x) - expm1(-x)) / 2
        let e_plus = self.exp_m1_p::<P>();
        let e_minus = (-self).exp_m1_p::<P>();
        (e_plus - e_minus) * Self::HALF
    }

    #[inline(always)]
    fn cosh_p<P: Policy>(self) -> Self {
        // (e^x + e^-x) / 2
        let e_plus = self.exp_p::<P>();
        let e_minus = (-self).exp_p::<P>();
        (e_plus + e_minus) * Self::HALF
    }

    #[inline(always)]
    fn tanh_p<P: Policy>(self) -> Self {
        // (e^2x - 1) / (e^2x + 1)
        let e2x_m1 = (self * Self::TWO).exp_m1_p::<P>();
        e2x_m1 / (e2x_m1 + Self::TWO)
    }

    #[inline(always)]
    fn asin_p<P: Policy>(self) -> Self {
        // asin(x) = atan(x / sqrt(1 - x^2))
        let one = Self::ONE;
        // (1-x)*(1+x) is generally more accurate than 1-x^2 near 1
        let omx2 = (one - self) * (one + self);
        let denom = omx2.sqrt();

        // if x=1, denom=0, atan approaches pi/2, handled correctly by atan2 ideally,
        // but simple division might return Inf. atan(Inf) = pi/2.
        // We use atan2 to handle the denom=0 case safely if needed, but atan_p handles Inf.

        // Check domain? if |x| > 1, omx2 is negative, sqrt is NaN.
        // Existing logic propagates NaN.
        (self / denom).atan_p::<P>()
    }

    #[inline(always)]
    fn acos_p<P: Policy>(self) -> Self {
        // acos(x) = pi/2 - asin(x)
        Self::FRAC_PI_2 - self.asin_p::<P>()
    }

    #[inline(always)]
    fn atan_p<P: Policy>(self) -> Self {
        let x = self;
        let abs_x = x.abs();

        // Constants
        // tan(pi/8) = sqrt(2) - 1
        let tan_pi_8 = Self::SQRT_2 - Self::ONE;

        // 1. Argument Reduction
        // Goal: reduce x to [0, tan(pi/8)] approx [0, 0.414]

        // Check if x > 1
        let gt_1 = abs_x.value().cmp_gt(Vector::ONE);

        // if x > 1: x = 1/x
        // We will compute pi/2 - atan(1/x) later
        let mut curr = gt_1.select(abs_x.reciprocal_p::<P>(), abs_x);

        // Check if x > tan(pi/8)
        let gt_tan_pi8 = curr.value().cmp_gt(tan_pi_8.value());

        // if x > tan(pi/8): x = (x-1)/(x+1)
        // We will add pi/4 later
        let shifted = (curr - Self::ONE) / (curr + Self::ONE);

        curr = gt_tan_pi8.select(shifted, curr);

        // 2. Series Evaluation
        // z - z^3/3 + z^5/5 ...

        let z = curr;
        let z2 = -(z * z); // negative for alternating series subtraction

        let mut sum = z;
        let mut term = z;

        let shift = if P::POLICY.unroll_loops { 2 } else { 0 };

        let mut i = 1;
        let max_i = (P::POLICY.max_iterations >> shift) + 1;

        // atan is very slow to converge
        while i < max_i {
            let next_i = i + (1 << shift);
            let prev = sum;

            for k in i..next_i {
                let div = (2 * k) + 1; // 3, 5, 7...

                term *= z2;

                // NOTE: Doesn't need explicit normalization later, due to sum being used
                sum.accumulate_unnormalized(term / Vector::splat(FloatElement::from_i64(div as i64)));
            }

            if prev.cmp_eq(sum).all() {
                // println!("atan converged at i={}", next_i - 1);
                break;
            }

            i = next_i;
        }

        // 3. Reconstruction

        // If we did the tan(pi/8) shift, add pi/4
        // sum = sum + pi/4
        sum = gt_tan_pi8.select(sum + Self::FRAC_PI_4, sum);

        // If we did the >1 inversion, subtract from pi/2
        // sum = pi/2 - sum
        sum = gt_1.select(Self::FRAC_PI_2 - sum, sum);

        // Restore Sign
        let xv = x.value();

        sum.value = sum.value.mul_sign(xv);
        sum.error = sum.error.mul_sign(xv);

        sum
    }

    #[inline(always)]
    fn atan2_p<P: Policy>(self, x: Self) -> Self {
        // y = self
        let y = self;
        let x_value = x.value();
        let zero = Self::ZERO;

        // Handle x = 0
        let x_is_zero = x_value.cmp_eq(Vector::ZERO);

        // If x=0, y>0 -> pi/2, y<0 -> -pi/2
        // We can cheat: atan2(y, 0) is roughly atan(Inf * sign(y))
        // But doing it explicitly is cleaner.

        let pi_2 = Self::FRAC_PI_2;
        let y_is_neg = y.value().cmp_lt(Vector::ZERO);
        let on_axis_res = y_is_neg.select(-pi_2, pi_2);

        // Standard case
        let z = y / x;
        let mut res = z.atan_p::<P>();

        // Adjust quadrant based on x and y
        // if x < 0:
        //   if y >= 0: res += pi
        //   if y < 0:  res -= pi

        let x_is_neg = x_value.cmp_lt(Vector::ZERO);
        let offset = Self::PI.conditional_negate(x_is_neg);

        res = x_is_neg.select(res + offset, res);

        x_is_zero.select(on_axis_res, res)
    }

    fn asinh_p<P: Policy>(self) -> Self {
        // ln(x + sqrt(x^2 + 1))
        // To avoid overflow for large x, use ln(2|x|) + ... or similar,
        // but for now direct implementation:
        // if x is negative, asinh(-x) = -asinh(x)

        let x_abs = self.abs();
        let y = x_abs + (x_abs * x_abs + Self::ONE).sqrt();
        let res = y.ln_p::<P>();

        let is_neg = self.value().cmp_lt(Vector::ZERO);
        is_neg.select(-res, res)
    }

    fn acosh_p<P: Policy>(self) -> Self {
        // ln(x + sqrt(x^2 - 1))
        // defined for x >= 1
        (self + (self * self - Self::ONE).sqrt()).ln_p::<P>()
    }

    fn atanh_p<P: Policy>(self) -> Self {
        // 0.5 * ln((1+x)/(1-x))
        let one = Self::ONE;
        let num = one + self;
        let den = one - self;
        (num / den).ln_p::<P>() * Self::HALF
    }

    #[inline(always)]
    fn exp_p<P: Policy>(self) -> Self {
        self.exp_internal_p::<P, EXP_MODE_EXP>()
    }

    #[inline(always)]
    fn exph_p<P: Policy>(self) -> Self {
        self.exp_internal_p::<P, EXP_MODE_EXPH>()
    }

    #[inline(always)]
    fn exp2_p<P: Policy>(self) -> Self {
        self.exp_internal_p::<P, EXP_MODE_POW2>()
    }

    #[inline(always)]
    fn exp10_p<P: Policy>(self) -> Self {
        self.exp_internal_p::<P, EXP_MODE_POW10>()
    }

    #[inline(always)]
    fn exp_m1_p<P: Policy>(self) -> Self {
        self.exp_internal_p::<P, EXP_MODE_EXPM1>()
    }

    #[inline(always)]
    fn powf_p<P: Policy>(self, e: Self) -> Self {
        (e * self.log2_p::<P>()).exp2_p::<P>()
    }

    #[inline(always)]
    fn cbrt_p<P: Policy>(self) -> Self {
        let s = MathWithPolicy::cbrt_p::<P>(self.value);
        let (p2, e2) = super::two_prod(s, s); // s^2
        let (p3, e3_base) = super::two_prod(p2, s); // s^3
        let e3 = s.mul_adde(e2, e3_base); // (s * e2) + e3_base
        let r = (self.value - p3) + (self.error - e3); // residual
        let deriv = Vector::<R>::splat(Element::from_i8(3)) * p2; // derivative = 3s^2

        Self::renormalized(s, r / deriv)
    }

    #[inline(always)]
    fn ln_p<P: Policy>(self) -> Self {
        self.log_internal_p::<P, LOG_MODE_LN>()
    }

    #[inline(always)]
    fn ln_1p_p<P: Policy>(self) -> Self {
        self.log_internal_p::<P, LOG_MODE_LN1P>()
    }

    #[inline(always)]
    fn log2_p<P: Policy>(self) -> Self {
        self.log_internal_p::<P, LOG_MODE_LOG2>()
    }

    #[inline(always)]
    fn log10_p<P: Policy>(self) -> Self {
        self.log_internal_p::<P, LOG_MODE_LOG10>()
    }

    fn log_p<P: Policy>(self, base: Self) -> Self {
        self.log2_p::<P>() / base.log2_p::<P>()
    }

    fn log_n_p<P: Policy, const N: usize>(self) -> Self {
        match N {
            0 => Self::ZERO,     // log(x)/log(0) = log(x)/-infinity = 0
            1 => Self::INFINITY, // log(x)/log(1) = log(x)/0 = complex infinity, only return real part
            2 => self.log2_p::<P>(),
            10 => self.log10_p::<P>(),
            n if n <= 32 => {
                // Use precomputed 1/ln(n) table for small integer bases
                let frac_1_ln_n: (R::Element, R::Element) =
                    <R::Element as CompensatedLogTable<R::Element>>::LOG_TABLE[n - 3];

                self.ln_p::<P>() * Self::from_parts(Vector::splat(frac_1_ln_n.0), Vector::splat(frac_1_ln_n.1))
            }
            _ => self.ln_p::<P>() / Vector::splat(FloatElement::from_i64(N as i64)).ln_p::<P>(),
        }
    }

    fn ln1m_expnx_p<P: Policy>(self) -> Self {
        self.ln1m_expnx_ext_p::<P>(self.ln_p::<P>())
    }

    fn ln1m_expnx_ext_p<P: Policy>(self, lnx: Self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn erf_p<P: Policy>(self) -> Self {
        self.erf_internal_p::<P>().0
    }

    #[inline(always)]
    fn erfc_p<P: Policy>(self) -> Self {
        self.erf_internal_p::<P>().1
    }

    #[inline(always)]
    fn erfinv_p<P: Policy>(self) -> Self {
        // High-performance erfinv using Halley's Method seeded by Winitzki's approximation.
        // This converges in ~3 iterations for 106-bit precision.
        // However, for very small |y|, we can do better with the Maclaurin series expansion,
        // despite more iterations, since it avoids expensive calls to log/exp/sqrt functions.

        let y = self;
        let abs_y = y.abs();
        let y_value = y.value();
        let abs_y_value = abs_y.value();

        if abs_y_value
            .cmp_le(const { Vector::splat_const(CompensatedElement::MAX_ERFINV_SERIES) })
            .all()
        {
            // For small |y|, use the Maclaurin series expansion for better performance,
            // since it doesn't need to call log/exp/sqrt/etc. functions.

            // Maclaurin series for erf_inv(y):
            // erf_inv(y) = sum_{k=0 to inf} (c_k / (2k+1)) * (sqrt(pi)/2 * y)^(2k+1)
            // where c_0 = 1, c_k = sum_{m=0 to k-1} (c_m * c_{k-1-m}) / ((m+1)(2m+1))

            let w = abs_y * Self::FRAC_SQRT_PI_2; // Variable w = (sqrt(pi)/2) * |y|
            let w2 = w * w;

            let mut sum = w; // Initial term (k=0): c_0 = 1, term = w
            let mut w_pow = w; // Stores w^(2k+1)

            const MAX_COEFFS: usize = 64;

            // Scalar Coefficient history buffer
            // We need this to compute the convolution for the next c_k.
            // 64 terms is generally sufficient for convergence where defined,
            // though it gets slow near |y| ~ 1.
            let mut coeffs = [Compensated::<R::Element>::ZERO; MAX_COEFFS];

            coeffs[0] = Compensated::ONE; // c_0 = 1

            let max_k = P::POLICY.max_iterations.min(MAX_COEFFS - 1);

            for k in 1..max_k {
                let prev = sum;

                let mut c_k = Compensated::ZERO;

                for m in 0..k {
                    // Term: (c_m * c_{k-1-m}) / ((m+1)(2m+1))
                    let num = coeffs[m] * coeffs[k - 1 - m];

                    let m_i = m as i64;
                    let den_i = (m_i + 1) * (2 * m_i + 1);

                    c_k.accumulate_unnormalized(num / Vector::splat(FloatElement::from_i64(den_i)));
                }

                coeffs[k] = c_k.normalize();

                // Term = (c_k / (2k+1)) * w^(2k+1)
                w_pow *= w2; // Next odd power of w

                let k_term_den = (2 * k + 1) as i64;

                let c_kv = Compensated {
                    value: Vector::splat(c_k.value.extract::<0>()),
                    error: Vector::splat(c_k.error.extract::<0>()),
                };

                sum.accumulate_unnormalized((c_kv * w_pow) / Vector::splat(FloatElement::from_i64(k_term_den)));

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
        let is_zero = abs_y_value.cmp_eq(Vector::ZERO);
        let is_one = abs_y_value.cmp_eq(Vector::ONE);

        // 1. Initial Guess via Winitzki's Approximation, using non-compensated math
        //    since the initial guess does not need to be incredibly accurate.
        // Relative error < 0.00035 across the domain.
        // Original: x ~ sqrt( sqrt(T1^2 - T2) - T1 )
        // Stable:   x ~ sqrt( -T2 / (sqrt(T1^2 - T2) + T1) )
        // This avoids catastrophic cancellation when y -> 0 (and thus T2 -> 0).

        // Constants
        let a = Vector::<R>::splat(FloatElement::from_f64(0.147)); // a = 0.147
        let c = Vector::FRAC_2_PI / a; // C = 2 / (pi * a)
        // L = ln(1 - y^2), use ln_1p for accuracy: ln(1 - y^2) = ln_1p(-y^2)
        let l = (-(y * y).value()).ln_1p_p::<P>();

        let half_l = l * Vector::HALF;
        let t1 = c + half_l;
        let t2 = l / a;

        // Stable Winitzki guess
        let root_term = t1.mul_sub(t1, t2).sqrt();
        let inner = -t2 / (root_term + t1);

        // Clamp inner to 0 to avoid NaN if y ~ 0 results in tiny negative due to noise
        let mut x = Self::new(inner.max(Vector::ZERO).sqrt());

        // 2. Halley's Method Iterations (Cubic Convergence)
        // x_{n+1} = x_n - u / (1 + x_n * u) where u = f(x_n) / f'(x_n)
        let tolerance = Self::tolerance_p::<P>().value();

        let skip = is_zero | is_one; // cannot be solved as roots

        for i in 0..P::POLICY.max_iterations {
            let f = x.erf_p::<P>() - abs_y; // Work with absolute y for stability

            // f / f'(x) = f * (sqrt(pi)/2) * exp(x^2)
            let u = f * (Self::FRAC_SQRT_PI_2 * (x * x).exp_p::<P>());

            // Halley step: u / (1 + x*u)
            // Note: f''/f' = -2x, so the Halley term simplifies to this.
            let step = u / x.mul_add(u, Vector::ONE);

            if (skip | step.value().abs().cmp_le(tolerance)).all() {
                // println!("erf_inv converged in Halley in {} iterations", i);
                break;
            }

            x.reduce_unnormalized(step);
        }

        x = is_zero.select(Self::ZERO, is_one.select(Self::INFINITY, x));

        // Restore sign: erf_inv(-y) = -erf_inv(y)
        x.value = x.value.mul_sign(y_value);
        x.error = x.error.mul_sign(y_value);

        x.normalize()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum ExpMode {
    Exp = 0,
    Expm1,
    Exph,
    Pow2,
    Pow10,
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum LogMode {
    Ln = 0,
    Log2,
    Log10,
    Ln1p,
}

const EXP_MODE_EXP: u8 = ExpMode::Exp as u8;
const EXP_MODE_EXPM1: u8 = ExpMode::Expm1 as u8;
const EXP_MODE_EXPH: u8 = ExpMode::Exph as u8;
const EXP_MODE_POW2: u8 = ExpMode::Pow2 as u8;
const EXP_MODE_POW10: u8 = ExpMode::Pow10 as u8;

const LOG_MODE_LN: u8 = LogMode::Ln as u8;
const LOG_MODE_LOG2: u8 = LogMode::Log2 as u8;
const LOG_MODE_LOG10: u8 = LogMode::Log10 as u8;
const LOG_MODE_LN1P: u8 = LogMode::Ln1p as u8;

impl<R: CompensatedRegister> Compensated<R>
where
    Vector<R>: MathWithPolicy<R>,
    R::Element: CompensatedRegister<Element = R::Element>,
{
    #[inline(always)]
    fn erf_internal_p<P: Policy>(self) -> (Self, Self) {
        let x = self;
        let abs_x = x.abs();

        let use_series = abs_x.value().cmp_lt(Vector::splat(FloatElement::from_f64(3.0))); // threshold can be tuned

        let use_only_series = use_series.all();
        let use_only_cf = use_series.none();

        // --- Init Series (erf) ---
        // erf(x) = 2/sqrt(pi) * (x - x^3/3 + x^5/10 ...)
        let x2 = -(x * x); // -x^2 for alternating series
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
            (true, false) => while i < max_i {
                let next_i = i + (1 << shift);
                let prev_s = sum_s;

                for k in i..next_i {
                    // --- Series Update ---
                    // term *= -x^2 * (2k-1) / (k * (2k+1))
                    let k_f = k as i64;
                    let k2_p1 = (2 * k + 1) as i64;
                    let k2_m1 = (2 * k - 1) as i64;

                    let num = FloatElement::from_i64(k2_m1);
                    let den = FloatElement::from_i64(k_f * k2_p1);

                    term_s *= x2 * (Compensated::splat(num) / Vector::splat(den));
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
                        a = Self::splat(FloatElement::from_i64((k - 1) as i64)) * Self::HALF;
                    }

                    // Lentz steps: D = b + a*D, C = b + a/C
                    d = a.mul_add(d, b); // D = b + a*D

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
            _ => while i < max_i {
                let next_i = i + (1 << shift);
                let prev_s = sum_s;
                let prev_f = f;

                for k in i..next_i {
                    // --- Series Update ---
                    // term *= -x^2 * (2k-1) / (k * (2k+1))
                    let k_f = k as i64;
                    let k2_p1 = (2 * k + 1) as i64;
                    let k2_m1 = (2 * k - 1) as i64;

                    let num = FloatElement::from_i64(k2_m1);
                    let den = FloatElement::from_i64(k_f * k2_p1);

                    // Compute ratio.
                    let s_ratio = x2 * (Compensated::splat(num) / Vector::splat(den));

                    term_s *= s_ratio; // overflow doesn't matter if we don't use this

                    let mut new_sum_s = sum_s;
                    new_sum_s.accumulate_unnormalized(term_s);

                    sum_s = use_series.select(new_sum_s, sum_s); // but avoid overflowing the sum

                    // --- CF Update ---
                    // Lentz coefficients: a_k = (k-1)/2
                    if k > 1 {
                        a = Self::splat(FloatElement::from_i64((k - 1) as i64)) * Self::HALF;
                    }

                    // Lentz steps: D = b + a*D, C = b + a/C
                    d = a.mul_add(d, b); // D = b + a*D

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
            f * (-x * x).exp_p::<P>() * Self::FRAC_1_SQRT_PI
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

        let final_erf = erf_val.conditional_negate(is_neg); // is_neg.select(-erf_val, erf_val);
        let final_erfc = is_neg.select(Self::TWO - erfc_val, erfc_val);

        (final_erf, final_erfc)
    }

    #[inline(always)]
    fn exp_internal_p<P: Policy, const MODE: u8>(mut self) -> Self {
        let mut is_inf = Mask::<R>::FALSY;
        let mut is_zero = Mask::<R>::FALSY;
        let mut is_invalid = Mask::<R>::FALSY;

        if const { P::POLICY.check_overflow } {
            let max_exp =
                <R::Element as FloatElement>::from_signed(<R::Element as FloatElement>::EXP_BIAS) + FloatConsts::ONE;

            let overflow_boundary = match MODE {
                EXP_MODE_POW2 => max_exp,
                EXP_MODE_POW10 => max_exp * FloatConsts::LOG10_2,
                EXP_MODE_EXP | EXP_MODE_EXPM1 | EXP_MODE_EXPH => max_exp * FloatConsts::LN_2,
                _ => FloatConsts::ZERO, // unreachable
            };

            is_inf = self.value.cmp_gt(Vector::splat(overflow_boundary));
            is_zero = self.value.cmp_lt(Vector::splat(-overflow_boundary)); // underflow to 0

            is_invalid = is_inf | is_zero;

            // replace these to avoid the loop producing NaNs/Infs/subnormals
            self = is_invalid.select(Self::ZERO, self);
        }

        let kf;
        let r;

        // range reduction
        if MODE == EXP_MODE_POW2 {
            // Base 2: 2^x, k = round(x), r = (x - k) * ln(2)
            kf = self.value.round();
            r = (self - Self::new(kf)) * Self::LN_2;
        } else if MODE == EXP_MODE_POW10 {
            // Base 10: 10^x = e^(x * ln10), k = round(x * log2(10)), r = x * ln(10) - k * ln(2)
            kf = (self.value * Vector::LOG2_10).round();
            // Standard Payne-Hanek style reduction step
            r = self.mul_sub(Self::LN_10, Self::new(kf) * Self::LN_2);
        } else {
            // Base e: exp(x), expm1(x), exph(x), k = round(x * log2(e)), r = x - k * ln(2)
            kf = (self.value * Vector::LOG2_E).round();
            r = Self::new(kf).mul_add(-Self::LN_2, self);
        }

        // start sum at 0 so that expm1 is accurate at k=0
        let mut sum = Self::ZERO;
        let mut term = Self::ONE;

        let shift = if P::POLICY.unroll_loops { 2 } else { 0 };

        let mut i = 1;
        let max_i = (P::POLICY.max_iterations >> shift) + 1;

        // Taylor series expansion
        while i < max_i {
            let next_i = i + (1 << shift);

            let prev_sum = sum;

            for i in i..next_i {
                term *= r / Vector::splat(FloatElement::from_i64(i as i64));
                sum.accumulate_unnormalized(term);
            }

            if prev_sum.cmp_eq(sum).all() {
                // println!("exp converged at i={}", next_i - 1);
                break;
            }

            i = next_i;
        }

        let mut k = kf.cast::<R::Signed>();

        // to divide res by 2, we can just subtract 1 from the exponent
        if MODE == EXP_MODE_EXPH {
            k -= Vector::ONE;
        }

        // sum + 1 also normalizes the compensated number
        let mut res = (sum + Self::ONE).ldexp_p::<P>(k); // 2^k * exp(r)

        if MODE == EXP_MODE_EXPM1 {
            res -= Self::ONE;

            res = k.cmp_eq(Vector::ZERO).select(sum, res);
        }

        if const { P::POLICY.check_overflow } {
            res.value = is_inf.select(Vector::INFINITY, res.value);
            res.value = is_zero.select(Vector::ZERO, res.value);

            // if either of the above, set error to 0
            res.error = is_invalid.select(Vector::ZERO, res.error);
        }

        res
    }

    #[inline(always)]
    fn log_internal_p<P: Policy, const MODE: u8>(mut self) -> Self {
        let mut is_nan = Mask::<R>::FALSY;
        let mut is_neg = Mask::<R>::FALSY;
        let mut is_zero = Mask::<R>::FALSY;
        let mut is_inf = Mask::<R>::FALSY;

        // 1. Domain Checks and Masking
        if const { P::POLICY.check_overflow } {
            // For standard logs: x < 0 is NaN, x == 0 is -Inf
            // For ln_1p: x < -1 is NaN, x == -1 is -Inf

            let zero_boundary = match MODE == LOG_MODE_LN1P {
                true => -Vector::ONE,
                false => Vector::ZERO,
            };

            is_nan = self.value.is_nan(); // Propagate existing NaNs
            is_inf = self.value.is_infinite() & self.value.cmp_gt(Vector::ZERO); // +Inf is valid

            let v = self.value();

            // Domain errors
            is_neg = v.cmp_lt(zero_boundary);
            is_zero = v.cmp_eq(zero_boundary);

            let is_invalid = is_nan | is_neg | is_zero | is_inf;

            // Sanitize: Replace invalid/Inf/Zero inputs with 1.0 (log(1) = 0)
            // This prevents the reduction step from producing garbage or panicking.
            self = is_invalid.select(Self::ONE, self);
        }

        // 2. Input Preparation
        // For ln_1p, we work on (self + 1).
        // Compensated arithmetic ensures that if 'self' is tiny, 'w' preserves it in the low part.
        let w = if MODE == LOG_MODE_LN1P { self + Self::ONE } else { self };

        // 3. Argument Reduction
        // We need x = m * 2^k, where m is in [0.5, 1.0)
        // We assume Vector/FloatElement supports a method to extract the exponent.
        // If not, this is usually: bit_cast_to_int(val) >> mantissa_bits - bias.

        let (mut m, mut k) = w.frexp_p::<P>();

        // 4. Domain Adjustment
        // The series for atanh(z) converges fastest when m is close to 1.
        // Standard frexp gives [0.5, 1). If m < 1/sqrt(2) (approx 0.707),
        // we double m and decrement k to push m into [0.707, 1.414].

        let needs_adj = m.value().cmp_lt(Vector::FRAC_1_SQRT_2);

        // if needs_adj: m *= 2, k -= 1
        m = needs_adj.select(m + m, m);
        k = needs_adj.select(k - Vector::ONE, k);

        // 5. Transform to z
        // ln(m) = 2 * atanh(z), where z = (m - 1) / (m + 1)

        let one = Self::ONE;
        let z = (m - one) / (m + one);
        let z_sq = z * z;

        // 6. Series Expansion: 2 * (z + z^3/3 + z^5/5 + ...)

        // Create initial term and sum based on mode,
        // this essentially distributes the multiplication ahead of time
        // for improved accuracy.
        let mut sum = match MODE {
            LOG_MODE_LOG2 => z * Self::LOG2_E,   // Convert ln to log2
            LOG_MODE_LOG10 => z * Self::LOG10_E, // Convert ln to log10
            _ => z,                              // natural log
        };

        let mut term = sum; // Current z^(2i+1)

        let shift = if P::POLICY.unroll_loops { 2 } else { 0 };
        let mut i = 1;

        // Note: logs converge very fast with this reduction.
        // 10-12 iters usually sufficient for 106-bit precision.
        let max_i = (P::POLICY.max_iterations >> shift) + 1;

        while i < max_i {
            let next_i = i + (1 << shift);

            let prev_sum = sum;

            for j in i..next_i {
                let div = (j * 2) + 1; // 3, 5, 7...

                term *= z_sq;
                sum.accumulate_unnormalized(term / Vector::splat(FloatElement::from_i64(div as i64)));
            }

            if prev_sum.cmp_eq(sum).all() {
                // println!("log converged at i={}", next_i - 1);
                break;
            }

            i = next_i;
        }

        // Multiply by 2 to complete 2*atanh(z)
        let ln_m = sum + sum; // also normalizes

        // We cast k back to Compensated to perform the final addition
        let k = Self::new(k.cast());

        // 7. Reconstruction: ln(x) = ln(m) + k * scale
        let mut res = match MODE {
            LOG_MODE_LOG2 => ln_m + k,
            LOG_MODE_LOG10 => ln_m + (k * Self::LOG10_2),
            _ => ln_m + (k * Self::LN_2),
        };

        // 9. Final Masking / Edge Cases
        if const { P::POLICY.check_overflow } {
            // Check original conditions

            // if x < 0 -> NaN
            res.value = is_neg.select(Vector::NAN, res.value);
            res.error = is_neg.select(Vector::NAN, res.error); // Error term of NaN is NaN

            // if x == 0 -> -Inf
            res.value = is_zero.select(Vector::NEG_INFINITY, res.value);
            res.error = is_zero.select(Vector::ZERO, res.error);

            // if x == +Inf -> +Inf
            res.value = is_inf.select(Vector::INFINITY, res.value);
            res.error = is_inf.select(Vector::ZERO, res.error);

            // if x was NaN -> NaN
            // (Often handled automatically by float math, but explicit select is safer)
            res.value = is_nan.select(Vector::NAN, res.value);
        }

        res
    }
}
