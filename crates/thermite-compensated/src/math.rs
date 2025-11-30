#![allow(unused)]

use super::{Compensated, CompensatedRegister};
use num_traits::MulAdd as _;
use thermite::{
    Mask, Vector,
    math::{
        FloatConsts, MathWithPolicy,
        policy::{Policy, PrecisionPolicy},
    },
    register::{Element, FloatElement, FloatRegister},
};

impl<R: CompensatedRegister> MathWithPolicy<R> for Compensated<R>
where
    Vector<R>: MathWithPolicy<R>,
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
                res = res.mul_add(self, Self::splat(c));
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
                res = res.mul_add(self, Self::splat(c));
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

    fn newtons_method_p<P: Policy, F>(self, tolerance: Self, bounds: Option<(Self, Self)>, f: F) -> Self
    where
        F: FnMut(Self) -> (Self, Self),
    {
        todo!()
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
                            res.mul_add(t, Self::splat(FloatElement::from_i64(c)))
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
        Self::ONE / self
    }

    #[inline(always)]
    fn inverse_sqrt_p<P: Policy>(self) -> Self {
        Self::ONE / self.sqrt()
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

    fn powiv_p<P: Policy>(self, e: Vector<R::Signed>) -> Self {
        todo!()
    }

    #[inline(always)]
    fn hypot_p<P: Policy>(self, y: Self) -> Self {
        self.mul_add(self, y * y).sqrt() // TODO: Check for overflow/underflow
    }

    fn sin_cos_p<P: Policy>(self) -> (Self, Self) {
        todo!()
    }

    fn sin_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn cos_p<P: Policy>(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn tan_p<P: Policy>(self) -> Self {
        self.sin_p::<P>() / self.cos_p::<P>()
    }

    #[inline(always)]
    fn sinc_p<P: Policy>(self) -> Self {
        self.sin_p::<P>() / self
    }

    #[inline(always)]
    fn sin_pix_p<P: Policy>(self) -> Self {
        self * (self * Self::PI).sin_p::<P>()
    }

    fn sinh_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn cosh_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn tanh_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn asin_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn acos_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn atan_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn atan2_p<P: Policy>(self, x: Self) -> Self {
        todo!()
    }

    fn asinh_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn acosh_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn atanh_p<P: Policy>(self) -> Self {
        todo!()
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
        (e * self.ln_p::<P>()).exp_p::<P>()
    }

    #[inline(always)]
    fn cbrt_p<P: Policy>(self) -> Self {
        let s = MathWithPolicy::cbrt_p::<P>(self.value);

        // s^2
        let (p2, e2) = super::two_prod(s, s);
        // s^3
        let (p3, e3_base) = super::two_prod(p2, s);

        let e3 = s.mul_adde(e2, e3_base); // (s * e2) + e3_base

        // residual
        let r = (self.value - p3) + (self.error - e3);

        // derivative = 3s^2
        let deriv = Vector::<R>::splat(Element::from_i8(3)) * p2;

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
        todo!()
    }

    fn log_n_p<P: Policy, const N: usize>(self) -> Self {
        todo!()
    }

    fn ln1m_expnx_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn ln1m_expnx_ext_p<P: Policy>(self, lnx: Self) -> Self {
        todo!()
    }

    fn erf_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn erfc_p<P: Policy>(self) -> Self {
        todo!()
    }

    fn erfinv_p<P: Policy>(self) -> Self {
        todo!()
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
{
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
            r = self.mul_add(Self::LN_10, Self::new(kf) * -Self::LN_2);
        } else {
            // Base e: exp(x), expm1(x), exph(x), k = round(x * log2(e)), r = x - k * ln(2)
            kf = (self.value * Vector::LOG2_E).round();
            r = Self::new(kf).mul_add(-Self::LN_2, self);
        }

        // start sum at 0 so that expm1 is accurate at k=0
        let mut sum = Self::ZERO;
        let mut term = Self::ONE;

        let threshold = Self::tolerance_p::<P>().value();

        let shift = if P::POLICY.unroll_loops { 2 } else { 0 };

        let mut i = 1;
        let max_i = (P::POLICY.max_iterations >> shift) + 1;

        // Taylor series expansion, 4 iterations at a time as a form of loop unrolling
        while i < max_i {
            let next_i = i + (1 << shift); // i + 4 if unrolling, else i + 1

            for i in i..next_i {
                term *= r / Compensated::splat(FloatElement::from_i64(i as i64));
                sum += term;
            }

            if term.value().abs().cmp_lt(threshold).all() {
                break;
            }

            i = next_i;
        }

        let mut k = kf.cast::<R::Signed>();

        // to divide res by 2, we can just subtract 1 from the exponent
        if MODE == EXP_MODE_EXPH {
            k -= Vector::ONE;
        }

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

        let mut sum = z;
        let mut term = z; // Current z^(2i+1)

        let threshold = Self::tolerance_p::<P>().value();

        let shift = if P::POLICY.unroll_loops { 2 } else { 0 };
        let mut i = 1;

        // Note: logs converge very fast with this reduction.
        // 10-12 iters usually sufficient for 106-bit precision.
        let max_i = (P::POLICY.max_iterations >> shift) + 1;

        while i < max_i {
            let next_i = i + (1 << shift);

            let mut last_term = term;

            for j in i..next_i {
                let div = (j * 2) + 1; // 3, 5, 7...

                term *= z_sq;
                last_term = term / Compensated::splat(FloatElement::from_i64(div as i64));
                sum += last_term;
            }

            // we track the last term added for convergence
            if last_term.value().abs().cmp_lt(threshold).all() {
                break;
            }

            i = next_i;
        }

        // Multiply by 2 to complete 2*atanh(z)
        let ln_m = sum + sum;

        // We cast k back to Compensated to perform the final addition
        let k = Self::new(k.cast());

        // 7. Reconstruction: ln(x) = ln(m) + k * scale
        let mut res = match MODE {
            LOG_MODE_LOG2 => ln_m.mul_add(Self::LOG2_E, k),
            LOG_MODE_LOG10 => ln_m.mul_add(Self::LOG10_E, k * Self::LOG10_2),
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
