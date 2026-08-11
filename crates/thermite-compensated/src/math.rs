use crate::{Compensated, CompensatedFloatVector, ScalarValue};

use thermite::prelude::*;

use thermite::element::FloatElementWithBits;
use thermite::vector::AsFloatVectorWithBitsKernel;

use thermite::math::policy::{PrecisionPolicy, policies::{CheckOverflow, PreserveDenormals}};
use thermite::math::specialized::{
    SpecializedCoreMath, SpecializedRealMath, SpecializedSpatialMath, SpecializedTranscendentalMath,
};
use thermite::math::{RealMathWithPolicy, TranscendentalMathWithPolicy};

impl<V: CompensatedFloatVector> SpecializedCoreMath<Compensated<V::Element>> for Compensated<V> {
    #[inline(always)]
    fn inverse_sqrt<P: Policy>(self) -> Self {
        Self::rsqrt(self)
    }
}

impl<V: CompensatedFloatVector> SpecializedTranscendentalMath<Compensated<V::Element>> for Compensated<V>
where
    V: TranscendentalMathWithPolicy,
{
    /// `$(\sin \pi x, \cos \pi x)$`, reducing **before** multiplying by pi.
    ///
    /// The inherited default is `sin_cos(self * PI)`, which throws away most of what
    /// this type exists for. Forming `x * PI` rounds the product, so the argument handed
    /// to `sin_cos` already carries an absolute error of about `|x| * 2^-106`; at
    /// `x = -1000.5` - an ordinary argument for the gamma reflection - that is three or
    /// four digits gone before any trigonometry happens.
    ///
    /// Reducing first avoids it entirely. `sin(pi(n + r)) = (-1)^n sin(pi r)` for integer
    /// `n`, and `x - round(x)` is *exact*, so the only rounded product is `r * PI` with
    /// `|r| <= 1/2`. Same for cosine, with the same sign flip.
    #[inline(always)]
    fn sincos_pi<P: Policy>(self) -> (Self, Self) {
        // n = round(x), r = x - n exactly, |r| <= 1/2.
        let n = self.value().round();
        let r = self - Self::new(n);

        let (s, c) = <Self as SpecializedTranscendentalMath<Compensated<V::Element>>>::sin_cos::<P>(r * Self::PI);

        // (-1)^n: odd n flips both. Halving is exact, so `n/2` having a fractional part
        // is the oddness test. Past 2^mantissa every representable value is even, which
        // this reports correctly rather than by accident.
        let half = n * V::HALF;
        let odd = half.cmp_ne(half.floor());

        (s.neg_c(odd), c.neg_c(odd))
    }

    #[inline(always)]
    fn sin_cos<P: Policy>(self) -> (Self, Self) {
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

        let r2 = -r.square(); // -r^2, used for iterative multiplication

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

                let div_c = (k2 - 1) * k2; // ((2k-1)*2k)
                let div_s = k2 * (k2 + 1); // (2k*(2k+1))

                // These should almost always succeed, since within the iteration limits
                // k will be at most around 20, so div_c and div_s will be at most
                // around 1600 or so before converging. This is well within the range of
                // even f32 integer representation.
                let (Some(div_c), Some(div_s)) = (FloatElement::try_from_int(div_c), FloatElement::try_from_int(div_s))
                else {
                    #[cold]
                    fn this_branch_is_unlikely() {}
                    this_branch_is_unlikely();
                    break;
                };

                // Update Cosine Term: prev_term * (-r^2) / div_c
                term_c *= r2 / V::splat(div_c);
                cos.accumulate_unnormalized(term_c);

                // Update Sine Term: prev_term * (-r^2) / div_s
                term_s *= r2 / V::splat(div_s);
                sin.accumulate_unnormalized(term_s);
            }

            if (prev_s.cmp_eq(sin) & prev_c.cmp_eq(cos)).all() {
                // println!("trig converged at i={}", next_i - 1);
                break;
            }

            i = next_i;
        }

        // If we have bit manipulation capabilities, we can analyze k directly
        // for efficient quadrant handling.
        if let Some((bit0, bit1)) = thermite::with_bits!([k]: [V; 1]
            as fn(values: [W; _]) -> (V::Mask, V::Mask) where V: CompensatedFloatVector
        {
            // Use integer mask logic on k. We can check the low 2 bits of k.
            let k_int: W::SignedBits = values[0].cast();

            // Construct bitmasks
            let bit0 = (k_int & NumericVector::ONE).cmp_ne(NumericVector::ZERO); // true if k % 2 != 0 (quadrants 1, 3)
            let bit1 = (k_int & NumericVector::TWO).cmp_ne(NumericVector::ZERO); // true if k % 4 >= 2 (quadrants 2, 3)

            (bit0.cast(), bit1.cast())
        }) {
            // Reconstruction using bitwise quadrant logic
            // Determine the final sin/cos based on the quadrant k.
            // The quadrant mapping for sin(x) / cos(x) where x = k * pi/2 + r:
            // k % 4 == 0:  sin ->  s,   cos ->  c
            // k % 4 == 1:  sin ->  c,   cos -> -s
            // k % 4 == 2:  sin -> -s,   cos -> -c
            // k % 4 == 3:  sin -> -c,   cos ->  s

            // Swap sin/cos if k is odd (quadrants 1, 3), and normalize the results
            let mut final_sin = bit0.select(cos, sin).normalize();
            let mut final_cos = bit0.select(sin, cos).normalize(); // Note: sign is handled next

            // Sign logic:
            // Sin sign: positive in 0, 1. Negative in 2, 3. -> invert if bit1 is true.
            // Cos sign: positive in 0, 3. Negative in 1, 2. -> invert if (bit0 ^ bit1) is true.
            let neg_sin = bit1;
            let neg_cos = bit0 ^ bit1;

            final_sin.value = final_sin.value.neg_c(neg_sin);
            final_sin.error = final_sin.error.neg_c(neg_sin);

            final_cos.value = final_cos.value.neg_c(neg_cos);
            final_cos.error = final_cos.error.neg_c(neg_cos);

            return (final_sin, final_cos); // skip the fallback implementation
        }

        // Reconstruction (Pure Float)
        //
        // We calculate coefficients S_k and C_k based on k mod 4 using only float math.
        // k_rem = k - 4 * round(k / 4). Range is {-2, -1, 0, 1, 2}.
        //
        // Mapping:
        // k_rem  |  C_k (cos k*pi/2)  |  S_k (sin k*pi/2)
        // ------------------------------------------------
        //   0    |    1               |    0
        //   1    |    0               |    1
        //   2    |   -1               |    0
        //  -1    |    0               |   -1
        //  -2    |   -1               |    0
        //
        // Formulas:
        // C_k = 1 - |k_rem|
        // S_k = k_rem * (2 - |k_rem|)

        let k_div4 = (k * V::splat(FloatElement::from_ratio(1, 4))).round();
        let k_rem = k_div4.nmul_adde(V::splat(<V::Element as FloatElement>::ConstInt::<4>::VALUE), k);
        let k_rem_abs = k_rem.abs();

        let c_k = V::ONE - k_rem_abs;
        let s_k = k_rem * (V::TWO - k_rem_abs);

        // Apply rotation:
        // sin(out) = sin(r)*C_k + cos(r)*S_k
        // cos(out) = cos(r)*C_k - sin(r)*S_k

        // Efficient mixing without full compensated addition (since terms are disjoint/zero)
        let final_sin = Compensated {
            value: cos.value.mul_adde(s_k, sin.value * c_k),
            error: cos.error.mul_adde(s_k, sin.error * c_k),
        };

        let final_cos = Compensated {
            value: sin.value.nmul_adde(s_k, cos.value * c_k),
            error: sin.error.nmul_adde(s_k, cos.error * c_k),
        };

        (final_sin, final_cos)
    }

    #[inline(always)]
    fn sinc<P: Policy>(self) -> Self {
        // Use non-compensated 4th root epsilon for tiny check, since
        // the Taylor series is actually very good for very small x.
        let is_tiny = self.value().abs().cmp_lt(FloatConsts::FOURTH_ROOT_EPSILON);

        let x2 = self.square();

        // if branching, use Taylor series for tiny x without calling sine.
        if !P::POLICY.avoid_branching && is_tiny.all() {
            let res = x2 / V::splat(<V::Element as FloatElement>::ConstInt::<120>::VALUE);
            return x2.mul_add(res - Self::FRAC_1_6, Self::ONE);
        }

        // For very small x, sinc(x) ~ 1 - x^2/6 + x^4/120
        let num = is_tiny.select(x2, self.sin_p::<P>());
        let den = is_tiny.select(
            Self::splat_value(<V::Element as FloatElement>::ConstInt::<120>::VALUE),
            self,
        );

        // combined division, since division is expensive
        let mut y = num / den;

        y = is_tiny.select(x2.mul_add(y - Self::FRAC_1_6, Self::ONE), y);

        if P::POLICY.check_overflow {
            y = self.value().is_infinite().select(Self::ZERO, y);
        }

        y
    }

    #[inline(always)]
    fn sinh_cosh<P: Policy>(self) -> (Self, Self) {
        let abs_x = self.abs();
        let ex = abs_x.exp_p::<P>();

        let hex_inv = Self::HALF / ex;
        let exh = Self::HALF * ex;

        // sinh = (e^x - e^-x) / 2, sinh is an odd function, so sinh(x) == -sinh(-x)
        // cosh = (e^x + e^-x) / 2, cosh is an even function, so cosh(x) == cosh(|x|)

        ((exh - hex_inv).mul_sign(self), exh + hex_inv)
    }

    #[inline(always)]
    fn sinh<P: Policy>(self) -> Self {
        // (e^x - e^-x) / 2
        let abs_x = self.abs();
        let ex = abs_x.exp_p::<P>();

        ex.mul_sube(Self::HALF, Self::HALF / ex).mul_sign(self)
    }

    #[inline(always)]
    fn cosh<P: Policy>(self) -> Self {
        // (e^x + e^-x) / 2
        // cosh is an even function, so cosh(x) == cosh(|x|)
        let abs_x = self.abs();
        let ex = abs_x.exp_p::<P>();

        ex.mul_adde(Self::HALF, Self::HALF / ex)
    }

    #[inline(always)]
    fn tanh<P: Policy>(self) -> Self {
        // (e^2x - 1) / (e^2x + 1)
        let e2x_m1 = (self + self).exp_m1_p::<P>();
        e2x_m1 / (e2x_m1 + Self::TWO)
    }

    #[inline(always)]
    fn asin<P: Policy>(self) -> Self {
        // asin(x) = atan(x / sqrt(1 - x^2))

        let omx2 = if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
            -self.square() + V::ONE // less accurate but faster
        } else {
            // (1-x)*(1+x) is generally more accurate than 1-x^2 near 1
            (Self::ONE - self) * (self + V::ONE)
        };

        (self / omx2.sqrt()).atan_p::<P>()
    }

    #[inline(always)]
    fn acos<P: Policy>(self) -> Self {
        // acos(x) = pi/2 - asin(x)
        Self::FRAC_PI_2 - self.asin_p::<P>()
    }

    #[inline(always)]
    fn atan<P: Policy>(self) -> Self {
        let x = self;
        let abs_x = x.abs();

        // Constants
        // tan(pi/8) = sqrt(2) - 1
        let tan_pi_8 = Self::SQRT_2 - Self::ONE;

        // 1. Argument Reduction
        // Goal: reduce x to [0, tan(pi/8)] approx [0, 0.414]

        // Check if x > 1
        let gt_1 = abs_x.value().cmp_gt(V::ONE);

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
        let z2 = -z.square(); // negative for alternating series subtraction

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
                sum.accumulate_unnormalized(term / V::splat(FloatElement::from_int(div as i64)));
            }

            if prev.cmp_eq(sum).all() {
                // println!("atan converged at i={}", next_i - 1);
                break;
            }

            i = next_i;
        }

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
    fn asinh<P: Policy>(self) -> Self {
        // asinh(x) = ln(x + sqrt(x^2 + 1)), rearranged so the argument never approaches 1.
        //
        // Written directly, small x sends `x + sqrt(x^2 + 1)` to 1 + x + O(x^2). A
        // double-double holds that to 106 bits *relative to 1*, so the part that carries
        // the answer keeps only 106 - log2(1/x) of them - at x = 1e-14 the result was good
        // to ~65 bits, not 106.
        //
        // With s = sqrt(1 + x^2), the offset from 1 is available in closed form:
        //   x + s - 1 = x + (s^2 - 1)/(s + 1) = x + x^2/(1 + s)
        // so feeding that to ln_1p keeps the small quantity small the whole way.

        let x_abs = self.abs();
        let s = (x_abs.square() + V::ONE).sqrt();
        let offset = x_abs.square() / (Self::ONE + s) + x_abs;

        // negate result if input was negative
        offset.ln_1p_p::<P>().neg_c(self.value().is_negative())
    }

    #[inline(always)]
    fn acosh<P: Policy>(self) -> Self {
        // ln(x + sqrt(x^2 - 1))
        // defined for x >= 1
        (self + (self.square() - V::ONE).sqrt()).ln_p::<P>()
    }

    #[inline(always)]
    fn atanh<P: Policy>(self) -> Self {
        // atanh(x) = 0.5 * ln((1+x)/(1-x)), through ln_1p for the same reason as `asinh`:
        // the ratio tends to 1 as x -> 0, and a double-double near 1 knows the part that
        // matters to only 106 - log2(1/x) bits.
        //
        //   (1 + x)/(1 - x) = 1 + 2x/(1 - x)
        //
        // so the offset from 1 is exact and small, and ln_1p costs the same as ln.
        let two_x = self + self;

        (two_x / (Self::ONE - self)).ln_1p_p::<P>() * Self::HALF
    }

    #[inline(always)]
    fn exp<P: Policy>(self) -> Self {
        Self::exp_internal::<P, EXP_MODE_EXP>(self)
    }

    #[inline(always)]
    fn exph<P: Policy>(self) -> Self {
        Self::exp_internal::<P, EXP_MODE_EXPH>(self)
    }

    #[inline(always)]
    fn exp2<P: Policy>(self) -> Self {
        Self::exp_internal::<P, EXP_MODE_POW2>(self)
    }

    #[inline(always)]
    fn exp10<P: Policy>(self) -> Self {
        Self::exp_internal::<P, EXP_MODE_POW10>(self)
    }

    #[inline(always)]
    fn exp_m1<P: Policy>(self) -> Self {
        Self::exp_internal::<P, EXP_MODE_EXPM1>(self)
    }

    #[inline(always)]
    fn exp2_m1<P: Policy>(self) -> Self {
        Self::exp_internal::<P, EXP_MODE_POW2M1>(self)
    }

    #[inline(always)]
    fn exp10_m1<P: Policy>(self) -> Self {
        Self::exp_internal::<P, EXP_MODE_POW10M1>(self)
    }

    #[inline(always)]
    fn powf<P: Policy>(self, e: Self) -> Self {
        // pow(x, y) = exp(y * ln(x))
        (e * self.ln_p::<P>()).exp_p::<P>()
    }

    #[inline(always)]
    fn cbrt<P: Policy>(self) -> Self {
        let s = self.value().cbrt_p::<P>();

        if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
            return Self::new(s);
        }

        // Compute s^3 using scalar 'two_prod'
        // This avoids full Compensated * Compensated overhead.
        // s^2 = p2 + e2
        let (p2, e2) = <V as ScalarValue>::square(s);
        // s^3 = p3 + e3_base (approx)
        let (p3, e3_base) = V::two_prod(p2, s);

        // Complete the error term for s^3: e3 = s*e2 + e3_base
        let e3 = s.mul_adde(e2, e3_base);

        // Compute High-Precision Residual: r = x - s^3
        // We perform the cancellation (value - p3) carefully.
        let diff_hi = self.value - p3;
        let diff_lo = self.error - e3;

        // Collapse to scalar (valid because diff is tiny, approx 10^-16)
        let r = diff_hi + diff_lo;

        // Halley Correction Term: s * (r / (2*s^3 + x))

        // Denom: 2*s^3 + x
        // We use p3 for s^3 (high part is sufficient for the denominator slope),
        // and avoid a register by just adding p3 to itself.
        let den = p3 + p3 + self.value;

        // Correction = s * (r / den)
        // We assume 'den' is safe (if self > 0).
        // If self == 0, den == 0, results in NaN (similar to your Newton code).
        let correction = s * (r / den);

        Self::renormalized(s, correction)
    }

    #[inline(always)]
    fn ln<P: Policy>(self) -> Self {
        // Initial Guess using base instruction
        let y_approx = Self::new(self.value().ln_p::<P>());

        if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
            return y_approx;
        }

        // Compute exp(y) with our high-precision implementation
        let e_y = y_approx.exp_p::<P>();

        // Halley's Iteration (One pass is sufficient for Double-Double)
        // Correction = 2 * (self - e_y) / (self + e_y)

        let diff = self - e_y;
        let sum = self + e_y;

        let correction = diff / sum;

        let result = y_approx + (correction + correction);

        // Handle Special Cases (Zero, Negative) if strictly required
        if const { P::POLICY.check_overflow } {
            // ln(0) -> -inf
            self.is_zero().select(Self::NEG_INFINITY, result)
        } else {
            result
        }
    }

    #[inline(always)]
    fn ln_1p<P: Policy>(self) -> Self {
        let y_approx = Self::new(self.value().ln_1p_p::<P>());

        if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
            return y_approx;
        }

        // We are solving: expm1(y) - x = 0
        //
        // Newton: y - f/f'
        // f(y) = e^y - 1 - x
        // f'(y) = e^y = (e^y - 1) + 1
        //
        // Halley: y - 2 * f * f' / (2(f')^2 - f * f'')
        // Since f'(y) = f''(y) = e^y, this simplifies nicely:
        //
        // Correction = 2 * (expm1(y) - x) / (expm1(y) + x + 2)

        let z = y_approx.exp_m1_p::<P>();

        let n = z - self;
        let d = z + self + Self::TWO;

        let correction = n / d;

        // Note: We subtract the correction because of the sign of the numerator (z - x).
        // Standard form is y - (f/...), here num is f, so we subtract.
        y_approx - (correction + correction)
    }

    #[inline(always)]
    fn log2<P: Policy>(self) -> Self {
        self.ln_p::<P>() * Self::LOG2_E
    }

    #[inline(always)]
    fn log10<P: Policy>(self) -> Self {
        self.ln_p::<P>() * Self::LOG10_E
    }

    #[inline(always)]
    fn log_n<P: Policy, const N: usize>(self) -> Self {
        match N {
            0 => Self::ZERO,     // log(x)/log(0) = log(x)/-infinity = 0
            1 => Self::INFINITY, // log(x)/log(1) = log(x)/0 = complex infinity, only return real part
            2 => self.log2_p::<P>(),
            10 => self.log10_p::<P>(),
            n if n <= 32 => {
                // Use precomputed 1/ln(n) table for small integer bases
                self.ln_p::<P>() * <V as crate::consts::CompensatedLogTable<V>>::LOG_TABLE[n - 3]
            }
            _ => self.ln_p::<P>() / V::splat(FloatElement::from_int(N as i64)).ln_p::<P>(),
        }
    }

    /// The `_ext` form exists so a caller who already has `ln(x)` can hand it to the
    /// low-precision approximation instead of paying for it twice. The compensated
    /// path never takes that approximation - it evaluates `ln(1 - e^-x)` exactly - so
    /// there is nothing to reuse and the hint is dropped, the same way the f64 kernel
    /// (`math/specialized/pd.rs`) and `Complex` do.
    #[inline(always)]
    fn ln1m_expnx_ext<P: Policy>(self, _lnx: Self) -> Self {
        self.ln1m_expnx_p::<P>()
    }
}

#[rustfmt::skip]
impl<V: CompensatedFloatVector> SpecializedSpatialMath<Compensated<V::Element>> for Compensated<V> {
    #[inline(always)] fn l2_norm_squared<P: Policy>(self) -> Self { self.square() }
    #[inline(always)] fn l2_norm<P: Policy>(self) -> Self { self.abs() }
    #[inline(always)] fn l1_norm<P: Policy>(self) -> Self { self.abs() }
}

impl<V: CompensatedFloatVector> SpecializedRealMath<Compensated<V::Element>> for Compensated<V>
where
    V: RealMathWithPolicy,
{
    #[inline(always)]
    fn atan2<P: Policy>(self, x: Self) -> Self {
        // y = self
        let y = self;
        let x_value = x.value();

        // Handle x = 0
        let x_is_zero = x_value.is_zero();

        // If x=0, y>0 -> pi/2, y<0 -> -pi/2
        // We can cheat: atan2(y, 0) is roughly atan(Inf * sign(y))
        // But doing it explicitly is cleaner.

        let pi_2 = Self::FRAC_PI_2;
        let y_is_neg = y.value().cmp_lt(V::ZERO);
        let on_axis_res = y_is_neg.select(-pi_2, pi_2);

        // Standard case
        let z = y / x;
        let mut res = z.atan_p::<P>();

        // Adjust quadrant based on x and y
        // if x < 0:
        //   if y >= 0: res += pi
        //   if y < 0:  res -= pi

        let x_is_neg = x_value.cmp_lt(V::ZERO);
        let offset = Self::PI.neg_c(x_is_neg);

        res = x_is_neg.select(res + offset, res);

        x_is_zero.select(on_axis_res, res)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum ExpMode {
    Exp = 0, // exp(x)
    Expm1,   // exp(x) - 1
    Exph,    // exp(x) / 2
    Pow2,    // 2^x
    Pow2m1,  // 2^x - 1
    Pow10,   // 10^x
    Pow10m1, // 10^x - 1
}

const EXP_MODE_EXP: u8 = ExpMode::Exp as u8;
const EXP_MODE_EXPM1: u8 = ExpMode::Expm1 as u8;
const EXP_MODE_EXPH: u8 = ExpMode::Exph as u8;
const EXP_MODE_POW2: u8 = ExpMode::Pow2 as u8;
const EXP_MODE_POW2M1: u8 = ExpMode::Pow2m1 as u8;
const EXP_MODE_POW10: u8 = ExpMode::Pow10 as u8;
const EXP_MODE_POW10M1: u8 = ExpMode::Pow10m1 as u8;

// impl<V: FloatVectorWithBits> Compensated<V> {
//     #[inline(always)]
//     pub(crate) fn ldexp_p<P: Policy>(self, exp: V::SignedBits) -> Self {
//         let value = self.value.ldexp_p::<P>(exp);
//         let error = self.error.ldexp_p::<P>(exp);
//         Self { value, error }
//     }

//     #[inline(always)]
//     pub(crate) fn frexp_p<P: Policy>(self) -> (Self, V::SignedBits) {
//         let (value, exp) = self.value.frexp_p::<P>();
//         let result = Self {
//             value,
//             error: self.error.ldexp_p::<P>(-exp), // scale error accordingly
//         };
//         (result, exp)
//     }
// }

const MAX_PRECISION_HI_ONLY: PrecisionPolicy = PrecisionPolicy::Average;

impl<V: CompensatedFloatVector> Compensated<V> {
    #[inline(always)]
    fn exp_core<P: Policy, const MODE: u8>(x: Self) -> Self {
        let mut sum = Self::ZERO;
        let mut term = Self::ONE;

        let mut i = 1;
        let shift = if P::POLICY.unroll_loops { 2 } else { 0 };
        let max_i = (P::POLICY.max_iterations >> shift) + i;

        // Taylor series expansion for expm1:
        while i < max_i {
            let next_i = i + (1 << shift);

            let prev_sum = sum;

            for k in i..next_i {
                let n = V::splat(FloatElement::from_int(k as i64));

                if const { P::POLICY.precision.gt(MAX_PRECISION_HI_ONLY) } {
                    term *= x / n;
                } else {
                    // using only r_hi for term calculation
                    term *= Self::from_fraction(x.value, n);
                }

                // accumulate_unnormalized is safe here because 'term' decreases in magnitude rapidly
                sum.accumulate_unnormalized(term);
            }

            if prev_sum.cmp_eq(sum).all() {
                // println!("exp converged at i={}", next_i - 1);
                break;
            }

            i = next_i;
        }

        // Note that the with_bits version will
        // need to do this with expm1, just not here.
        if const { EXP_MODE_EXPM1 != MODE && EXP_MODE_POW2M1 != MODE && EXP_MODE_POW10M1 != MODE } {
            sum += V::ONE;
        }

        if const { P::POLICY.precision.le(MAX_PRECISION_HI_ONLY) } {
            // Linear correction for low part
            // sum += exp(r_hi) * r_lo
            if const { EXP_MODE_EXPM1 == MODE || EXP_MODE_POW2M1 == MODE || EXP_MODE_POW10M1 == MODE } {
                sum = (sum + V::ONE).mul_adde(x.error, sum);
            } else {
                sum = sum.mul_adde(x.error, sum);
            }
        }

        sum
    }

    #[inline(always)]
    fn exp_internal<P: Policy, const MODE: u8>(x: Self) -> Self {
        struct ExpKernelWithBits<V: CompensatedFloatVector, P: Policy, const MODE: u8>(
            core::marker::PhantomData<(V, P)>,
        );

        // V _might_ have the ability to do bit-manipulation, but we don't know that. This is
        // a way to access a type `W` that has the same bits as `V`, but implements
        // the necessary traits for bit-level operations. If that's the case, we can use better
        // range-reduction techniques and opt for ldexp instead of repeated squaring.
        impl<V: CompensatedFloatVector, P: Policy, const MODE: u8> AsFloatVectorWithBitsKernel<V, 2>
            for ExpKernelWithBits<V, P, MODE>
        {
            type Output = Compensated<V>;

            #[inline(always)]
            fn with_bits<
                W: FloatVectorWithBits<
                        Element = V::Element,
                        Lanes = V::Lanes,
                        Mask = V::Mask,
                        Signed = V::Signed,
                        Unsigned = V::Unsigned,
                        ExtendedPrecision = <V as FloatVector>::ExtendedPrecision,
                    > + CastVector<V>,
            >(
                self,
                c: [W; 2],
            ) -> Self::Output {
                let mut x = Compensated::<V> {
                    value: c[0].cast_into(),
                    error: c[1].cast_into(),
                };

                let mut overflows: V::Mask = GenericMask::FALSY;
                let mut underflows: V::Mask = GenericMask::FALSY;

                if const { P::POLICY.check_overflow } {
                    let max_exp = <W::Element as FloatElementWithBits>::from_signed(
                        <W::Element as FloatElementWithBits>::EXP_BIAS,
                    ) + Element::ONE;

                    let overflow_boundary = match MODE {
                        EXP_MODE_POW2 | EXP_MODE_POW2M1 => max_exp,
                        EXP_MODE_POW10 | EXP_MODE_POW10M1 => max_exp * FloatConsts::LOG10_2,
                        EXP_MODE_EXP | EXP_MODE_EXPM1 | EXP_MODE_EXPH => max_exp * FloatConsts::LN_2,
                        _ => Element::ZERO, // unreachable
                    };

                    overflows = x.value.cmp_gt(V::splat(overflow_boundary));
                    underflows = x.value.cmp_lt(V::splat(-overflow_boundary)); // underflow to 0

                    // zero the lanes to avoid the loop producing NaNs/Infs/subnormals
                    x = x.nz(overflows | underflows);
                }

                let k;
                let mut r;

                // range reduction
                if const { EXP_MODE_POW2 == MODE || EXP_MODE_POW2M1 == MODE } {
                    // Base 2: 2^x, k = round(x), r = (x - k) * ln(2)
                    k = x.value().round();
                    r = (x - k) * Compensated::LN_2;
                } else {
                    if const { EXP_MODE_POW10 == MODE || EXP_MODE_POW10M1 == MODE } {
                        // Base 10: 10^x = e^(x * ln10), k = round(x * log2(10)), r = x * ln(10) - k * ln(2)
                        k = (x.value() * <V as FloatConsts>::LOG2_10).round();

                        r = x * Compensated::LN_10;
                    } else {
                        // Base e: exp(x), expm1(x), exph(x), k = round(x * log2(e)), r = x - k * ln(2)
                        k = (x.value() * <V as FloatConsts>::LOG2_E).round();

                        r = x;
                    }

                    // Standard 3-part Payne-Hanek style reduction
                    r -= k * V::LN_2_EXTENDED[0];
                    r -= k * V::LN_2_EXTENDED[1];
                    r -= k * V::LN_2_EXTENDED[2];
                }

                // we only need k as an integer for ldexp
                let mut k = k.cast::<W>().cast::<W::SignedBits>();

                // exp_core returns either exp or expm1 of the reduced argument
                let mut y = Compensated::<V>::exp_core::<P, MODE>(r);

                let y0 = y; // save for exm1 adjustment

                if const { EXP_MODE_EXPH == MODE } {
                    // to divide res by 2, we can just subtract 1 from the exponent
                    k -= NumericVector::ONE;
                } else if const { EXP_MODE_EXPM1 == MODE || EXP_MODE_POW2M1 == MODE || EXP_MODE_POW10M1 == MODE } {
                    // expm1/exp2m1/exp10m1 needs an adjustment of +1 before scaling
                    y += V::ONE;
                }

                // 2^k * exp(r), go through W WithBits type for ldexp
                // don't bother with overflow checks in ldexp, we've already done that
                y.value = W::cast_from(y.value).ldexp_p::<CheckOverflow<P, false>>(k).cast_into();
                // The low word gets `Preserve`. The overflow pre-check above bounds the
                // *value*, which is what justifies scaling it with the exponent clamp
                // turned off - but it says nothing about the low word, which sits ~53
                // binades below and leaves the representable range first. An unclamped
                // `ldexp` writes a negative biased exponent straight into the exponent
                // field: exp(-700) came back with a value of 9.86e-305 and a low word of
                // -2.74e+295, which then poisoned everything refining through `exp`
                // (`ln(1e-300)` was off by exactly 2.0, `ln(1e300)` was NaN, because
                // Halley's `(x - e_y)/(x + e_y)` collapses to -1 on a garbage `e_y`).
                //
                // `PreserveDenormals` takes `ldexp`'s two-multiply path, which lets IEEE
                // gradual underflow produce the subnormal instead of wrapping - so the
                // correction survives rather than merely not being poison.
                y.error = W::cast_from(y.error)
                    .ldexp_p::<PreserveDenormals<CheckOverflow<P, false>>>(k)
                    .cast_into();

                // Backstop: a correction can never be as large as the value it corrects.
                // Cannot fire on a well-formed result, and costs one compare.
                y.error = y.error.nz(y.error.abs().cmp_ge(y.value.abs()));

                if const { EXP_MODE_EXPM1 == MODE || EXP_MODE_POW2M1 == MODE || EXP_MODE_POW10M1 == MODE } {
                    // small input values get the raw unscaled result
                    y = k.is_zero().select(y0, y - V::ONE);
                }

                if const { P::POLICY.check_overflow } {
                    // zero lane on underflow, set to infinity on overflow
                    y = overflows.select(Compensated::INFINITY, y.nz(underflows));
                }

                y
            }
        }

        // This will always be zero-cost, but will only succeed if V
        // supports the necessary bit-level operations, returning None otherwise.
        if let Some(res) = V::with_bits(
            [x.value, x.error],
            ExpKernelWithBits::<V, P, MODE>(core::marker::PhantomData),
        ) {
            return res; // skip the fallback implementation
        }

        // approximate scaling factor based on element size
        // f32 = 8, f64 = 12
        let n = size_of::<V::Element>() + 4;

        // crude range-reduction, assumes x is not larger than 2^N,
        // which is reasonable for exp inputs.
        let scale = V::splat(FloatElement::from_int(1 << n));
        let mut r = x / scale;

        let overflows = r.value.cmp_gt(V::ONE);
        let underflows = r.value.cmp_lt(V::NEG_ONE);

        r = r.nz(overflows | underflows); // zero lane on overflow/underflow

        if const { EXP_MODE_POW2 == MODE || EXP_MODE_POW2M1 == MODE } {
            r *= Self::LN_2;
        } else if const { EXP_MODE_POW10 == MODE || EXP_MODE_POW10M1 == MODE } {
            r *= Self::LN_10;
        }

        let mut y = Self::exp_core::<P, MODE>(r);

        for _ in 0..n {
            let y_sq = y.square();

            y = if const { EXP_MODE_EXPM1 == MODE || EXP_MODE_POW2M1 == MODE || EXP_MODE_POW10M1 == MODE } {
                // correction for expm1 squaring
                y_sq + Compensated {
                    value: y.value + y.value, // 2x should be lossless
                    error: y.error + y.error, // using addition for performance
                }
            } else {
                y_sq
            };
        }

        if const { EXP_MODE_EXPH == MODE } {
            y *= Self::HALF;
        }

        // zero lane on underflow, set to infinity on overflow
        y = overflows.select(Self::INFINITY, y.nz(underflows));

        y
    }
}
