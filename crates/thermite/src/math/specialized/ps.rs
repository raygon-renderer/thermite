use crate::{
    divider::Divider,
    math::policy::policies::MediumPrecision,
    vector::ops::{AddMasked as _, MulAddExt as _},
};
use core::f32::consts::{FRAC_1_PI, FRAC_PI_2, LN_10, LOG2_E, SQRT_2};

use super::*;

impl<V: FloatVectorWithBits<Element = f32>> SpecializedCoreMath<f32> for V {
    #[inline(always)]
    fn inverse_sqrt<P: Policy>(self) -> Self {
        super::generic::inverse_sqrt_internal::<V, f32, P>(self)
    }
}

#[rustfmt::skip]
impl<V: FloatVectorWithBits<Element = f32>> SpecializedSpatialMath<f32> for V {
    #[inline(always)] fn l2_norm_squared<P: Policy>(self) -> Self { self * self }
    #[inline(always)] fn l2_norm<P: Policy>(self) -> Self { self.abs() }
    #[inline(always)] fn l1_norm<P: Policy>(self) -> Self { self.abs() }
}

impl<V: FloatVectorWithBits<Element = f32>> SpecializedTranscendentalMath<f32> for V {
    #[inline(always)]
    fn sinc<P: Policy>(self) -> Self {
        super::generic::sinc_internal::<V, f32, P>(self)
    }

    #[inline(always)]
    fn sinc_pi<P: Policy>(self) -> Self {
        super::generic::sinc_pi_internal::<V, f32, P>(self)
    }

    #[inline(always)]
    fn log_n<P: Policy, const N: usize>(self) -> Self {
        super::generic::log_n_internal::<V, f32, P, N>(self)
    }

    #[inline(always)]
    fn sin_cos<P: Policy>(self) -> (Self, Self) {
        if const {
            P::POLICY.precision.le(PrecisionPolicy::Average)
                && Self::NATIVE_CAP.has(NativeCapability::SIN | NativeCapability::COS)
        } {
            return unsafe { self.native_sin_cos::<P>() };
        }

        sin_cos_f_internal::<P, V, false, false>(self)
    }

    #[inline(always)]
    fn sin<P: Policy>(self) -> Self {
        if const { P::POLICY.precision.le(PrecisionPolicy::Average) && Self::NATIVE_CAP.has(NativeCapability::SIN) } {
            return unsafe { self.native_sin::<P>() };
        }

        sin_cos_f_internal::<P, V, false, true>(self).0
    }

    #[inline(always)]
    fn cos<P: Policy>(self) -> Self {
        if const { P::POLICY.precision.le(PrecisionPolicy::Average) && Self::NATIVE_CAP.has(NativeCapability::COS) } {
            return unsafe { self.native_cos::<P>() };
        }

        sin_cos_f_internal::<P, V, false, true>(self).1
    }

    #[inline(always)]
    fn tan<P: Policy>(self) -> Self {
        if const { P::POLICY.precision.le(PrecisionPolicy::Average) && Self::NATIVE_CAP.has(NativeCapability::TAN) } {
            return unsafe { self.native_tan::<P>() };
        }

        let d = self;

        // Instead of computing sin(x)/cos(x) (which suffers from catastrophic cancellation near pi/2),
        // this uses a direct polynomial approximation for tan on [-pi/4, pi/4] and handles odd
        // quadrants via negation + reciprocal (i.e. -cot(x) = -1/tan(x)).

        let xa = d.abs().flush_denormals::<P>();

        let (mut x, mut x_lo, q) = trig_range_reduction::<P, V, false>(xa);

        // For odd quadrants (q & 1 == 1), negate x before the polynomial.
        // Combined with reciprocal at the end, this gives -cot(x) = -1/tan(x).
        let odd_sign = V::from_bits(q.shli::<31>());
        x ^= odd_sign;
        x_lo ^= odd_sign;

        // Polynomial: tan(x) ~= x + x^3 * P(x^2)
        // Minimax coefficients for (tan(x)/x - 1) / x^2 on [-pi/4, pi/4]
        let x2 = x * x;
        let mut x0 = x;

        if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            x0 += x_lo;
        }

        #[rustfmt::skip]
        let mut r = x2.poly_rev_p::<P, _>(&[
            9.38540185543E-3,   // x^12
            3.11992232697E-3,   // x^10
            2.44301354525E-2,   //  x^8
            5.34112807005E-2,   //  x^6 : ~17/315
            1.33387994085E-1,   //  x^4 : ~2/15
            3.33331568548E-1,   //  x^2 : ~1/3
        ]).mul_adde(x2 * x, x0);

        // For odd quadrants, take reciprocal: 1/tan(-x) = -1/tan(x) = -cot(x)
        let odd = (q & V::Bits::ONE).cmp_ne(V::Bits::ZERO);

        if const { P::POLICY.avoid_branching } || odd.any() {
            r = odd.select(r.reciprocal_p::<P>(), r);
        }

        // Apply sign of original input (tan is an odd function)
        r = r.mul_sign(d);

        if const { P::POLICY.check_overflow } {
            // tan(±inf) = NaN, tan(NaN) = NaN
            r = d.is_finite().select(r, V::NAN);
        }

        r
    }

    #[inline(always)]
    fn sincos_pi<P: Policy>(self) -> (Self, Self) {
        if const {
            P::POLICY.precision.le(PrecisionPolicy::Average)
                && Self::NATIVE_CAP.has(NativeCapability::SIN | NativeCapability::COS)
        } {
            return unsafe { (self * Self::PI).native_sin_cos::<P>() };
        }

        sin_cos_f_internal::<P, V, true, false>(self)
    }

    #[inline(always)]
    fn sin_pi<P: Policy>(self) -> Self {
        if const { P::POLICY.precision.le(PrecisionPolicy::Average) && Self::NATIVE_CAP.has(NativeCapability::SIN) } {
            return unsafe { (self * Self::PI).native_sin::<P>() };
        }

        sin_cos_f_internal::<P, V, true, true>(self).0
    }

    #[inline(always)]
    fn cos_pi<P: Policy>(self) -> Self {
        if const { P::POLICY.precision.le(PrecisionPolicy::Average) && Self::NATIVE_CAP.has(NativeCapability::COS) } {
            return unsafe { (self * Self::PI).native_cos::<P>() };
        }

        sin_cos_f_internal::<P, V, true, true>(self).1
    }

    #[inline(always)]
    fn tan_pi<P: Policy>(self) -> Self {
        if const { P::POLICY.precision.le(PrecisionPolicy::Average) && Self::NATIVE_CAP.has(NativeCapability::TAN) } {
            return unsafe { (self * Self::PI).native_tan::<P>() };
        }

        let (s, c) = self.sincos_pi::<P>();
        s / c
    }

    #[inline(always)]
    fn sinh_cosh<P: Policy>(self) -> (Self, Self) {
        let x0 = self;
        let x = x0.abs().flush_denormals::<P>();
        let y = x.exph_p::<P>();
        let qy = V::FRAC_1_4 / y;

        let mut sinh = y - qy;
        let cosh = y + qy;

        let x_small = x.cmp_lt(V::ONE);

        // if any are small, use a polynomial approximation
        if const { P::POLICY.precision.ge(PrecisionPolicy::Average) } && (P::POLICY.avoid_branching || x_small.any()) {
            let x2 = x * x;

            let y1 = x2
                .poly_rev_p::<P, _>(&[2.03721912945E-4, 8.33028376239E-3, 1.66667160211E-1])
                .mul_adde(x2 * x, x);

            sinh = x_small.select(y1, sinh);
        }

        (sinh.mul_sign(x0), cosh)
    }

    #[inline(always)]
    fn sinh<P: Policy>(self) -> Self {
        let x0 = self;
        let x = x0.abs().flush_denormals::<P>();

        let x_small = x.cmp_lt(V::ONE);

        let mut y2 = V::EMPTY;

        // if not all are small, use exponential functions. Tiers with
        // `precision < Average` skip the small-x polynomial below, so they must
        // run this path unconditionally - otherwise all-small input leaves
        // `y2 == 0` and `sinh(small)` returns 0.
        if const { P::POLICY.avoid_branching || P::POLICY.precision.lt(PrecisionPolicy::Average) } || !x_small.all() {
            y2 = x.exph_p::<P>();
            y2 -= V::FRAC_1_4 / y2;

            if const { P::POLICY.avoid_precision_branches() } {
                return y2.mul_sign(x0);
            }
        }

        // if any are small, use a polynomial approximation
        if const { P::POLICY.precision.ge(PrecisionPolicy::Average) } && (P::POLICY.avoid_branching || x_small.any()) {
            let x2 = x * x;

            let y1 = x2
                .poly_rev_p::<P, _>(&[2.03721912945E-4, 8.33028376239E-3, 1.66667160211E-1])
                .mul_adde(x2 * x, x);

            y2 = x_small.select(y1, y2);
        }

        y2.mul_sign(x0)
    }

    #[inline(always)]
    fn cosh<P: Policy>(self) -> Self {
        let y = self.abs().exph_p::<P>();
        y + V::FRAC_1_4 / y
    }

    #[inline(always)]
    #[rustfmt::skip]
    fn tanh<P: Policy>(self) -> Self {
        let x0 = self;
        let one = V::ONE;

        let x = x0.abs().flush_denormals::<P>();
        let x_small = x.cmp_lt(crate::const_splat!(f32: 0.625));

        let mut y2 = V::EMPTY;

        // if not all are small. Tiers with `precision < Average` skip the
        // small-x polynomial below, so they must run this path unconditionally
        // (else all-small input leaves `y2 == 0` and `tanh(small)` returns 0).
        if const { P::POLICY.avoid_branching || P::POLICY.precision.lt(PrecisionPolicy::Average) } || !x_small.all() {
            // tanh(x) = (e^2x - 1) / (e^2x + 1). `exph` returns e^t / 2 with one
            // extra bit of exponent headroom, so with h = exph(2x) = e^2x / 2
            // the identity folds to (h - 1/2) / (h + 1/2): same value, e^2x
            // overflows slightly later, and no extra square is needed.
            // (Note `exph(x)^2` would be e^2x / 4, which is *not* what tanh
            // wants - that was the bug here.)
            let h = (x + x).exph_p::<P>();
            y2 = (h - V::HALF) / (h + V::HALF);

            if const { P::POLICY.check_overflow } {
                y2 = x.cmp_gt(crate::const_splat!(f32: 44.4)).select(one, y2);
            }

            if const { P::POLICY.avoid_precision_branches() } {
                return y2.mul_sign(x0);
            }
        }

        // if any are small
        if const { P::POLICY.precision.ge(PrecisionPolicy::Average) } && (P::POLICY.avoid_branching || x_small.any()) {
            let x2 = x * x;

            let y1 = x2.poly_rev_p::<P, _>(&[
                -5.70498872745E-3,
                2.06390887954E-2,
                -5.37397155531E-2,
                1.33314422036E-1,
                -3.33332819422E-1,
            ]).mul_adde(x2 * x, x);

            y2 = x_small.select(y1, y2);
        }

        y2.mul_sign(x0)
    }

    #[inline(always)]
    fn asin<P: Policy>(self) -> Self {
        asin_f_internal::<P, Self, false>(self)
    }

    #[inline(always)]
    fn acos<P: Policy>(self) -> Self {
        asin_f_internal::<P, Self, true>(self)
    }

    #[inline(always)]
    fn atan<P: Policy>(self) -> Self {
        let x = self;
        let t = x.abs().flush_denormals::<P>();

        if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
            /* http://mathforum.org/library/drmath/view/62672.html
             * Examined 4278190080 values of atan:
             *   2.36864877 avg ULP diff, 302 max ULP, 6.55651e-06 max error      // (with  denormals)
             * Examined 4278190080 values of atan:
             *   171160502 avg ULP diff, 855638016 max ULP, 6.55651e-06 max error // (crush denormals)
             */
            let a = t;

            let gt1 = a.cmp_gt(V::ONE);

            let s = gt1.select(a.reciprocal_p::<ExtraPrecision<P>>().flush_denormals::<P>(), a);

            let t = s * s;

            // place the s * 0.43157974 in the FMA to encourage instruction-level parallelism
            let r = t.mul_adde(s * crate::const_splat!(f32: 0.43157974), s)
                / t.mul_adde(
                    crate::const_splat!(f32: 0.05831938),
                    crate::const_splat!(f32: 0.76443945),
                )
                .mul_adde(t, V::ONE);

            let r = gt1.select(V::FRAC_PI_2 - r, r);

            return r.copysign(x);
        }

        let not_small = t.cmp_ge(crate::const_splat!(f32: SQRT_2 - 1.0)); // t >= tan  pi/8
        let not_big = t.cmp_le(crate::const_splat!(f32: SQRT_2 + 1.0)); // t <= tan 3pi/8

        let s = not_big.select(V::FRAC_PI_4, V::FRAC_PI_2);

        // small:  z = t / 1.0;
        // medium: z = (t-1.0) / (t+1.0);
        // big:    z = -1.0 / t;

        // lightweight select logic using zeroing and conditional adds
        let a = V::NEG_ONE.zz(not_small).add_c(not_big, t);
        let b = V::ONE.zz(not_big).add_c(not_small, t);

        let z = a / b;
        let z2 = z * z;

        z2.poly_rev_p::<P, _>(&[8.05374449538E-2, -1.38776856032E-1, 1.99777106478E-1, -3.33329491539E-1])
            .mul_adde(z2 * z, z.add_c(not_small, s)) // z += select(not_small, s, 0.0);
            .mul_sign(x)
    }

    #[inline(always)]
    fn asinh<P: Policy>(self) -> Self {
        let x0 = self;

        let x = x0.abs().flush_denormals::<P>();
        let x2 = x * x;

        let x_small = x.cmp_le(crate::const_splat!(f32: 0.51));

        let mut y2 = V::EMPTY;

        if const { P::POLICY.avoid_branching } || !x_small.all() {
            let x21 = if const { V::HAS_TRUE_FMA } {
                x.mul_add(x, V::ONE)
            } else {
                x2 + V::ONE
            };

            y2 = (x21.sqrt() + x).ln_p::<P>();

            if const { P::POLICY.check_overflow } {
                let x_huge = x.cmp_gt(crate::const_splat!(f32: 1e10));

                if const { P::POLICY.avoid_precision_branches() } || crate::unlikely(x_huge.any()) {
                    y2 = x_huge.select(x.ln_p::<P>() + V::LN_2, y2);
                }
            }

            if const { P::POLICY.avoid_precision_branches() } {
                return y2.mul_sign(x0);
            }
        }

        if const { P::POLICY.avoid_branching } || x_small.any() {
            let y1 = x2
                .poly_rev_p::<P, _>(&[2.0122003309E-2, -4.2699340972E-2, 7.4847586088E-2, -1.6666288134E-1])
                .mul_adde(x2 * x, x);

            y2 = x_small.select(y1, y2);
        }

        y2.mul_sign(x0)
    }

    #[inline(always)]
    fn acosh<P: Policy>(self) -> Self {
        let x0 = self.flush_denormals::<P>();
        let x1 = x0 - V::ONE;

        let x_small = x1.cmp_lt(crate::const_splat!(f32: 0.49)); // use Pade approximation if abs(x-1) < 0.5

        let mut y2 = V::EMPTY;

        // if not all are small
        if const { P::POLICY.avoid_branching } || !x_small.all() {
            y2 = (x0.mul_sube(x0, V::ONE).sqrt() + x0).ln_p::<P>();

            if const { P::POLICY.check_overflow } {
                let x_huge = x1.cmp_gt(crate::const_splat!(f32: 1e10));

                if const { P::POLICY.avoid_precision_branches() } || crate::unlikely(x_huge.any()) {
                    y2 = x_huge.select(x0.ln_p::<P>() + V::LN_2, y2);
                }
            }

            if const { P::POLICY.avoid_precision_branches() } {
                return y2;
            }
        }

        // if any are small
        if const { P::POLICY.avoid_branching } || x_small.any() {
            #[rustfmt::skip]
            let mut y1 = x1.sqrt() * x1.poly_rev_p::<P, _>(&[
                1.7596881071E-3,
                -7.5272886713E-3,
                2.6454905019E-2,
                -1.1784741703E-1,
                1.4142135263E0,
            ]);

            if const { P::POLICY.check_overflow } {
                // result is NaN if less-than 1
                y1 = x0.cmp_lt(V::ONE).select(V::NAN, y1);
            }

            y2 = x_small.select(y1, y2);
        }

        y2
    }

    #[inline(always)]
    fn atanh<P: Policy>(self) -> Self {
        let x = self.abs().flush_denormals::<P>();

        let x_small = x.cmp_lt(V::HALF);

        let mut y2 = V::EMPTY;

        if const { P::POLICY.avoid_branching } || !x_small.all() {
            let one = V::ONE;

            y2 = ((one + x) / (one - x)).ln_p::<P>().scale(0.5);

            if const { P::POLICY.check_overflow } {
                let y3 = x.cmp_eq(one).select(V::INFINITY, V::NAN);
                y2 = x.cmp_ge(one).select(y3, y2);
            }

            if const { P::POLICY.avoid_precision_branches() } {
                return y2.mul_sign(self);
            }
        }

        if const { P::POLICY.avoid_branching } || x_small.any() {
            let x2 = x * x;

            #[rustfmt::skip]
            let y1 = x2.poly_rev_p::<P, _>(&[
                1.81740078349E-1,
                8.24370301058E-2,
                1.46691431730E-1,
                1.99782164500E-1,
                3.33337300303E-1,
            ])
            .mul_adde(x2 * x, x);

            y2 = x_small.select(y1, y2);
        }

        y2.mul_sign(self)
    }

    #[inline(always)]
    fn exp<P: Policy>(self) -> Self {
        exp_f_internal::<P, Self, EXP_MODE_EXP>(self)
    }

    #[inline(always)]
    fn exph<P: Policy>(self) -> Self {
        exp_f_internal::<P, Self, EXP_MODE_EXPH>(self)
    }

    #[inline(always)]
    fn exp2<P: Policy>(self) -> Self {
        exp_f_internal::<P, Self, EXP_MODE_POW2>(self)
    }

    #[inline(always)]
    fn exp10<P: Policy>(self) -> Self {
        exp_f_internal::<P, Self, EXP_MODE_POW10>(self)
    }

    #[inline(always)]
    fn exp_m1<P: Policy>(self) -> Self {
        exp_f_internal::<P, Self, EXP_MODE_EXPM1>(self)
    }

    #[inline(always)]
    fn exp2_m1<P: Policy>(self) -> Self {
        exp_f_internal::<P, Self, EXP_MODE_POW2M1>(self)
    }

    #[inline(always)]
    fn exp10_m1<P: Policy>(self) -> Self {
        exp_f_internal::<P, Self, EXP_MODE_POW10M1>(self)
    }

    #[inline(always)]
    fn powf<P: Policy>(self, y: Self) -> Self {
        if const { P::POLICY.precision.le(PrecisionPolicy::Average) && Self::NATIVE_CAP.has(NativeCapability::POWF) } {
            return unsafe { self.native_powf::<P>(y) };
        }

        let x0 = self;
        let y = y.flush_denormals::<P>();

        if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
            // the "Worst" log2 precision is _terrible_, so just use medium
            // to give anything reasonable back
            return (x0.log2_p::<MediumPrecision<P>>() * y).exp2_p::<P>();
        }

        // define constants
        let ln2f_hi: V = crate::const_splat!(f32: 0.693359375); // log(2), split in two for extended precision
        let ln2f_lo: V = crate::const_splat!(f32: -2.12194440e-4);
        let log2e = V::LOG2_E;
        let ln2 = V::LN_2;

        let zero = V::ZERO;
        let one = V::ONE;
        let half = V::HALF;

        let x1 = x0.abs().flush_denormals::<P>();

        let mut x = fraction2::<V>(x1);

        let blend = x.cmp_gt(crate::const_splat!(f32: SQRT_2 * 0.5));

        // reduce range of x = +/- sqrt(2)/2
        x.add_assign_c(!blend, x); // conditional assign, only if blend is false
        x -= one;

        // Taylor expansion, high precision
        let x2 = x * x;

        // logarithm expansion
        let mut lg1 = x.poly_rev_p::<P, _>(&[
            7.0376836292E-2,
            -1.1514610310E-1,
            1.1676998740E-1,
            -1.2420140846E-1,
            1.4249322787E-1,
            -1.6668057665E-1,
            2.0000714765E-1,
            -2.4999993993E-1,
            3.3333331174E-1,
        ]);

        lg1 *= x2 * x;

        let ef = V::cast_from(exponent::<V>(x1)).add_c(blend, one);

        // multiply exponent by y, nearest integer e1 goes into exponent of result, remainder yr is added to log
        let e1 = (ef * y).round();
        let yr = ef.mul_sube(y, e1); // calculate remainder yr. precision very important here

        // add initial terms to expansion
        let lg = half.nmul_adde(x2, x) + lg1; // lg = (x - 0.5f * x2) + lg1;

        // calculate rounding errors in lg
        // rounding error in multiplication 0.5*x*x
        let x2err = (half * x).mul_sube(x, half * x2);

        // rounding error in additions and subtractions
        let lgerr = half.mul_adde(x2, lg - x) - lg1; // lgerr = ((lg - x) + 0.5f * x2) - lg1;

        // extract something for the exponent
        let e2 = (lg * y * log2e).round();

        // subtract this from lg, with extra precision
        let mut v = e2.nmul_adde(ln2f_lo, lg.mul_sube(y, e2 * ln2f_hi));

        // correct for previous rounding errors
        v -= (lgerr + x2err).mul_sube(y, yr * ln2);

        // extract something for the exponent if possible
        let mut x = v;
        let e3 = (x * log2e).round();

        // high precision multiplication not needed here because abs(e3) <= 1
        x = e3.nmul_adde(ln2, x); // x -= e3 * float(VM_LN2);

        let x2 = x * x;

        // Taylor expansion of exp
        let z = x
            .poly_rev_p::<P, _>(&[1.0 / 5040.0, 1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 1.0 / 2.0])
            .mul_adde(x2, x + one);

        // contributions to exponent
        let ee = e1 + e2 + e3;
        let ei: V::SignedBits = ee.fast_cast();

        // biased exponent of result:
        let ej = ei + (V::SignedBits::from_bits(z.abs()) >> 23);

        // add exponent by signed integer addition
        let mut z = V::from_bits(V::SignedBits::from_bits(z) + (ei << 23));

        if const { !P::POLICY.check_overflow } {
            // x^0 == 1 is important enough to keep even on the fast path (the
            // exponent-split form otherwise leaves x's exponent in for y == 0).
            return y.cmp_eq(zero).select(one, z);
        }

        // check exponent for overflow and underflow
        let overflow =
            ej.cmp_ge(V::SignedBits::splat(0x0FF)).cast::<V::Mask>() | ee.cmp_gt(crate::const_splat!(f32: 300.0));
        let underflow =
            ej.cmp_le(V::SignedBits::splat(0x000)).cast::<V::Mask>() | ee.cmp_lt(crate::const_splat!(f32: -300.0));

        // check for special cases
        let xfinite = x0.is_finite();
        let yfinite = y.is_finite();
        let efinite = ee.is_finite();

        let xzero = x0.is_zero_or_subnormal();
        let xsign = x0.is_negative();

        z = underflow.select(zero, z);
        z = overflow.select(V::INFINITY, z);

        let yzero = y.cmp_eq(zero);
        let yneg = y.cmp_lt(zero);

        // pow_case_x0
        z = xzero.select(yneg.select(V::INFINITY, yzero.select(one, zero)), z);

        let mut yodd = zero;

        if xsign.any() {
            let yint = y.cmp_eq(y.round());
            yodd = V::from_bits(y.into_bits::<V::Bits>() << 31);

            let z0 = x0.cmp_eq(zero).select(z, V::NAN);
            let z1 = yint.select(z | yodd, z0);

            yodd = yint.select(yodd, zero);

            z = xsign.select(z1, z);
        }

        let not_special = (xfinite & yfinite & (efinite | xzero));

        if crate::likely(not_special.all()) {
            return z; // fast return
        }

        // handle special error cases: y infinite
        let z1 = (yfinite & efinite).select(
            z,
            x1.cmp_eq(one)
                .select(one, (x1.cmp_gt(one) ^ y.is_negative()).select(V::INFINITY, zero)),
        );

        // handle x infinite
        let z1 = xfinite.select(
            z1,
            yzero.select(
                one,
                yneg.select(
                    yodd & z, // 0.0 with the sign of z from above
                    // x1 | (x0 & yodd), // get sign of x0 only if y is odd integer
                    V::ternlog::<{ crate::ternlog_imm!(A | (B & C)) }>(x1, x0, yodd),
                ),
            ),
        );

        // Always propagate nan:
        // Deliberately differing from the IEEE-754 standard which has pow(0,nan)=1, and pow(1,nan)=1
        (x0.is_nan() | y.is_nan()).select(x0 + y, z1)
    }

    #[inline(always)]
    fn cbrt<P: Policy>(self) -> Self {
        let x = self.flush_denormals::<P>();

        let b1: V::Bits = crate::const_splat!(u32: 709958130); // B1 = (127-127.0/3-0.03306235651)*2**23
        let b2: V::Bits = crate::const_splat!(u32: 642849266); // B2 = (127-127.0/3-24/3-0.03306235651)*2**23
        let m: V::Bits = crate::const_splat!(u32: 0x7fffffff); // u32::MAX >> 1

        let x1p24 = x * crate::const_splat!(f32: f32::from_bits(0x4b800000)); // 0x1p24f === 2 ^ 24

        let hx0: V::Bits = x.into_bits::<V::Bits>() & m;

        let x_small = hx0.cmp_lt(crate::const_splat!(u32: 0x00800000));

        let xs = x_small.select(x1p24, x);
        let b = x_small.select(b2, b1);

        let mut ui: V::Bits = xs.into_bits();
        let mut hx = ui & m;

        // NOTE: Using the branched divider with a constant
        // leads to better codegen when the branch is inlined.
        hx = hx / Divider::u32(3) + b;

        ui &= V::Bits::splat(0x80000000);
        ui |= hx;

        let mut t = V::from_bits(ui);

        // using extended precision is slower but perfectly accurate, but the single-precision
        // branch is only remotely accurate with fused multiply-adds.
        if const { P::POLICY.precision.ge(PrecisionPolicy::Best) || !Self::HAS_TRUE_FMA } {
            let mut td: Self::ExtendedPrecision = t.cast();
            let xd: Self::ExtendedPrecision = x.cast();

            // First iteration accurate to 16 bits, second iteration to 47 bits.
            for _ in 0..2 {
                let r = td * td * td;
                let rxd = xd + r;
                td *= (xd + rxd) / (r + rxd);
            }

            t = td.cast();
        } else {
            let two = V::TWO;

            // couple iterations of Halley's method
            // This isn't perfect, as it's only limited to single-precision,
            // but the fused multiply-adds helps
            for _ in 0..2 {
                let t3 = t * t * t;
                t *= two.mul_add(x, t3) / two.mul_add(t3, x); // try to use extended precision where possible
            }

            // FMA residual correction - compute t^3 - x precisely, then one Newton step
            if const { P::POLICY.precision.ge(PrecisionPolicy::Average) } {
                let t2 = t * t;
                t -= t2.mul_sub(t, x) / (t2 * crate::const_splat!(f32: 3.0)); // t^3 - x, exact to FMA precision
            }
        }

        if const { !P::POLICY.check_overflow } {
            return x.cmp_eq(V::ZERO).select(x, t);
        }

        // cbrt(NaN,INF,+-0) is itself
        (hx0.cmp_gt(V::Bits::splat(0x7f800000)) | hx0.cmp_eq(V::Bits::ZERO)).select(x, t)
    }

    #[inline(always)]
    fn ln<P: Policy>(self) -> Self {
        ln_f_internal::<P, Self, false>(self)
    }

    #[inline(always)]
    fn ln_1p<P: Policy>(self) -> Self {
        ln_f_internal::<P, Self, true>(self)
    }

    #[inline(always)]
    fn log2<P: Policy>(self) -> Self {
        ln_2_internal::<P, Self>(self)
    }

    #[inline(always)]
    fn log10<P: Policy>(self) -> Self {
        ln_10_internal::<P, Self>(self)
    }

    #[inline(always)]
    fn ln1m_expnx<P: Policy>(self) -> Self {
        let x = self;

        if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
            return x.ln1m_expnx_ext_p::<P>(x.ln_p::<P>());
        }

        (V::ONE - (-x).exp_p::<P>()).ln_p::<P>()
    }

    #[inline(always)]
    fn ln1m_expnx_ext<P: Policy>(self, lnx: Self) -> Self {
        let x = self;

        if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
            let x = x.flush_denormals::<P>();

            // determined empirically
            const X1: f32 = 9.1;
            const X2: f32 = 16.3;

            const B: f32 = 1.0 / (X2 - X1); // b
            const AB: f32 = X1 / (X2 - X1); // a*b where a=x1

            // combined into fma
            //let u1 = (x - V::splat(x1)) * V::splat(1.0 / (x2 - x1));
            let u1 = x.mul_sube(crate::const_splat!(f32: B), crate::const_splat!(f32: AB));

            // clamp
            let mut u1 = u1.min(V::ONE).max(V::ZERO);

            if const { P::POLICY.precision.eq(PrecisionPolicy::Medium) } {
                u1 = u1.smoothstep_p::<P, 2>(None);
            }

            // ResourceFunction["MiniMaxApproximation"][Log[x] - Log[1 - Exp[-x]], {x, {0.01, 20.0}, 3, 5}]
            // let c = x.poly_p::<P, _>(&[-0.000165121, 0.501311, 0.0308712, 0.0123851])
            //     / x.poly_p::<P, _>(&[1.0, 0.149063, 0.0346305, 0.00306313, -0.0000128591]);

            // ResourceFunction["MiniMaxApproximation"][Log[x] - Log[1 - Exp[-x]], {x, {0.01, 20.0}, 5, 7}]
            let c = x.poly_rational_p::<P, _, _>(
                &[0.0, 0.5, 0.0439145, 0.0116566, 0.000713523, 0.0000392684],
                &[
                    1.0,
                    0.171161,
                    0.0375791,
                    0.0038616,
                    0.000283035,
                    7.93625e-6,
                    -1.02103e-8,
                    7.10327e-12,
                ],
            );

            // bring to zero on the tail
            let mut res = u1.lerp_p::<P>(lnx - c, V::ZERO);

            if const { P::POLICY.check_overflow } {
                res = res.cmp_lt(V::ZERO).select(V::NAN, res);
                res = res.cmp_eq(V::ZERO).select(V::NEG_INFINITY, res);
            }

            return res;
        }

        (V::ONE - (-x).exp_p::<P>()).ln_p::<P>()
    }
}

impl<V: FloatVectorWithBits<Element = f32>> SpecializedRealMath<f32> for V {
    #[inline(always)]
    fn wrap_angle<P: Policy>(self) -> Self {
        let x = self;
        let n = ((x + Self::PI) * (Self::FRAC_1_PI * Self::HALF)).floor();

        if const { Self::HAS_TRUE_FMA || P::POLICY.precision.le(PrecisionPolicy::Average) } {
            return n.nmul_adde(Self::TAU, x);
        }

        // Cody-Waite: split TAU so n * tau_hi is exact
        let tau_hi: V = crate::const_splat!(f32: hexf::hexf32!("0x1.921fb60000000p+2"));
        let tau_lo: V = crate::const_splat!(f32: hexf::hexf32!("-0x1.777a5c0000000p-23"));
        (x - n * tau_hi) - n * tau_lo
    }

    #[inline(always)]
    fn atan2<P: Policy>(self, x: Self) -> Self {
        let y = self;
        let neg_one = V::NEG_ONE;
        let zero = V::ZERO;

        let x1 = x.abs().flush_denormals::<P>();
        let y1 = y.abs().flush_denormals::<P>();

        let swap_xy = y1.cmp_gt(x1);

        if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
            let (a, b) = (x1, y1);

            let n = swap_xy.select(b, a);
            let d = swap_xy.select(a, b);

            let mut k = n / d;

            if const { P::POLICY.check_overflow } {
                let b_eq_zero = b.cmp_eq(V::ZERO);
                let ab_eq = a.cmp_eq(b);

                k = ab_eq.select(V::ONE, k);
                k = b_eq_zero.select(V::ZERO, k);
            }

            let s = k.flush_denormals::<P>();

            let t = s * s;

            let mut r = t.mul_adde(s * crate::const_splat!(f32: 0.43157974), s)
                / t.mul_adde(
                    crate::const_splat!(f32: 0.05831938),
                    crate::const_splat!(f32: 0.76443945),
                )
                .mul_adde(t, V::ONE);

            r = swap_xy.select(V::FRAC_PI_2 - r, r);
            r = x.select_negative(V::PI - r, r);

            return r.copysign(y);
        }

        let mut x2 = swap_xy.select(y1, x1);
        let mut y2 = swap_xy.select(x1, y1);

        if const { P::POLICY.check_overflow } {
            let both_infinite = x.is_infinite() & y.is_infinite();

            //if crate::unlikely(both_infinite.any())
            x2 = both_infinite.select(x2 & neg_one, x2); // get 1.0 with the sign of x
            y2 = both_infinite.select(y2 & neg_one, y2); // get 1.0 with the sign of y
        }

        // x = y = 0 will produce NAN. No problem, fixed below
        let t = y2 / x2;

        // small:  z = t / 1.0;
        // medium: z = (t-1.0) / (t+1.0);
        let not_small = t.cmp_ge(crate::const_splat!(f32: SQRT_2 - 1.0));

        let a = t + neg_one.zz(not_small);
        let b = V::ONE + t.zz(not_small);

        let s = V::FRAC_PI_4.zz(not_small);

        let z = a / b;
        let z2 = z * z;

        let mut re = z2
            .poly_rev_p::<P, _>(&[8.05374449538E-2, -1.38776856032E-1, 1.99777106478E-1, -3.33329491539E-1])
            .mul_adde(z2 * z, z + s);

        re = swap_xy.select(V::FRAC_PI_2 - re, re);
        re = (x | y).is_zero().select(zero, re); // atan2(0,+0) = 0 by convention
        re = x.select_negative(V::PI - re, re); // also for x = -0.

        re.copysign(y)
    }
}

#[thermite_macros::dispatch(V, thermite = "crate")]
fn payne_hanek_reduction<P: Policy, V: FloatVectorWithBits<Element = f32>>(xa: &V) -> (V, V, V::Bits) {
    let xa_bits: V::Bits = xa.into_bits();

    // Extract unbiased exponent and significand
    let exp = (V::SignedBits::from_bits(xa_bits.shri::<23>()) & V::SignedBits::splat(0xFF)) - V::SignedBits::splat(127);
    let exp_u: V::Unsigned = V::Bits::from_bits(exp.max(V::SignedBits::ZERO)).cast();

    // 24-bit significand with implicit hidden bit restored
    let sig = (xa_bits & V::Bits::splat(0x007FFFFF)) | V::Bits::splat(0x00800000);

    // Padded 2/pi table: one zero word prepended to absorb the -26 offset.
    // Index with (exp + 6) instead of (exp - 26) to avoid unsigned underflow.
    const INVPI_TABLE: [u32; 7] = [
        0x00000000, // padding
        0xA2F9836E, 0x4E441529, 0xFC2757D1, 0xF534DDC0, 0xDB629599, 0x3C439041,
    ];

    let biased = exp_u + V::Unsigned::splat(6); // always >= 6, never underflows
    let idx: V::Unsigned = biased.shri::<5>();
    let shift = biased & V::Unsigned::splat(31);
    let inv_shift = (V::Unsigned::splat(32) - shift) & V::Unsigned::splat(31);

    let c0 = unsafe { V::Unsigned::lookup_unchecked(&INVPI_TABLE, idx) };
    let c1 = unsafe { V::Unsigned::lookup_unchecked(&INVPI_TABLE, idx + V::Unsigned::ONE) };
    let c2 = unsafe { V::Unsigned::lookup_unchecked(&INVPI_TABLE, idx + V::Unsigned::TWO) };

    // Shift chunks to align binary point
    // Mask shifts by 31 to prevent UB on shift == 32 in some ISAs
    let mask = shift.cmp_ne(V::Unsigned::ZERO);
    let aligned_hi = c0.shlv(shift) | c1.shrv(inv_shift).zz(mask);
    let aligned_lo = c1.shlv(shift) | c2.shrv(inv_shift).zz(mask);

    let aligned_hi: V::Bits = aligned_hi.cast();
    let aligned_lo: V::Bits = aligned_lo.cast();

    // Multiply significand by aligned chunks.
    // 88-bit product: sig(24) * aligned(64).
    // Binary point at bit 62: bits 62:61 = quadrant, bits 60:0 = fraction.
    let prod_hi = sig.mullo(aligned_hi); // bits 63:32 (low half of sig * hi)
    let prod_lo = sig.mulhi(aligned_lo); // bits 55:32 (high half of sig * lo)
    let mid_bits = prod_hi + prod_lo; // bits 63:32 of the 88-bit product
    let prod_lo_lo = sig.mullo(aligned_lo); // bits 31:0

    // Extract quadrant from bits 30:29 of mid_bits.
    let mut q_ph: V::Bits = (mid_bits.shri::<29>()) & V::Bits::splat(3);

    // 61-bit fraction: 29 bits from mid_bits (bits 28:0) + 32 bits from prod_lo_lo.
    let fraction_hi_int = mid_bits & V::Bits::splat(0x1FFFFFFF);

    // Reconstruct as double-float (two non-overlapping f32 values).
    //
    // frac_hi: top 23 bits of fraction_hi_int, injected as f32 mantissa (exact).
    // Represents (fraction_hi_int >> 6) * 2^-23.
    let frac_hi_bits = fraction_hi_int.shri::<6>() | V::Bits::splat(0x3F800000);
    let frac_hi = V::from_bits(frac_hi_bits) - V::ONE;

    // frac_lo: bottom 6 bits of fraction_hi_int | top 18 bits of prod_lo_lo = 24 bits.
    // Represents residual * 2^-47. Exact since residual <= 2^24 - 1.
    let residual = (fraction_hi_int & V::Bits::splat(0x3F)).shli::<18>() | prod_lo_lo.shri::<14>();
    let frac_lo_int: V::SignedBits = residual.cast();
    let frac_lo = V::cast_from(frac_lo_int) * crate::const_splat!(f32: f32::from_bits(0x28000000)); // 2^-47

    // Center from [0, 1) to [-0.5, 0.5) to match Cody-Waite's round().
    // Only frac_hi needs adjustment; frac_lo is unchanged since
    // (frac_hi - 1) + frac_lo = old_total - 1.
    let needs_round = frac_hi.cmp_ge(V::HALF);
    let frac_hi = frac_hi.sub_c(needs_round, V::ONE);
    q_ph = q_ph.add_c(needs_round.cast(), V::Bits::ONE);

    // Multiply by π/2 as double-float.
    // π/2 = pi2_hi + pi2_lo where pi2_hi = f32(π/2) and pi2_lo = π/2 - f32(π/2).
    let pi2_hi = V::FRAC_PI_2;
    let pi2_lo = crate::const_splat!(f32: -4.37113882867379288655e-08);

    let x_hi = frac_hi * pi2_hi;

    // Recover rounding error via exact FMA, then add cross terms.
    // frac_lo * pi2_lo is O(2^-71), negligible.
    //
    // Note that if hardware FMA is not available, this will be much slower
    // but that's just the cost of accuracy.
    let x_lo = frac_hi.mul_add(pi2_hi, -x_hi) + frac_hi * pi2_lo + frac_lo * pi2_hi;

    (x_hi, x_lo, q_ph)
}

/// Shared Cody-Waite range reduction for single-precision trig functions.
///
/// Reduces `xa` (absolute value, flushed) modulo pi/2, returning (x_hi, x_lo, quadrant_bits).
/// `x_lo` is nonzero only when Payne-Hanek is used (large args, Best+ precision).
/// When `PI` is true, performs sinpi/cospi reduction instead (no CW, no Payne-Hanek).
#[inline(always)]
fn trig_range_reduction<P: Policy, V: FloatVectorWithBits<Element = f32>, const PI: bool>(
    mut xa: V,
) -> (V, V, V::Bits) {
    let mut is_large = V::Mask::FALSY;

    let y0 = if PI {
        xa + xa // 2x for sinpi/cospi
    } else {
        is_large = xa.cmp_gt(crate::const_splat!(<V> = <V: FloatVector> f32: {
            match V::HAS_TRUE_FMA {
                true => 1e7,
                false => 1e5,
            }
        }));

        if const { P::POLICY.check_overflow && P::POLICY.precision.le(PrecisionPolicy::Average) } {
            xa = xa.nz(is_large); // set to zero if too large
        }

        xa.scale(FloatConsts::FRAC_2_PI)
    };

    let y = y0.round();
    let mut q: V::Bits = V::SignedBits::fast_cast_from(y).into_bits();

    // pi/2 split into four parts for extended precision modular arithmetic.
    // dp1 (7 sig bits) + dp2 (10 sig bits) + dp3 (10 sig bits) + dp4 (10 sig bits) = pi/4.
    // All constants are doubled since we reduce by pi/2, not pi/4.
    let mut x = if PI {
        // sinpi/cospi: x = pi * (xa - y * 0.5)
        y.nmul_adde(V::HALF, xa).scale(FloatConsts::PI)
    } else if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
        // Single-step reduction: xa - y * (pi/2). One FMA, no extended precision.
        // Loses ~7 bits relative to the full Cody-Waite, acceptable at Medium.
        y.nmul_adde(V::FRAC_PI_2, xa)
    } else {
        let dp1f = crate::const_splat!(f32: 0.78515625 * 2.0);
        let dp2f = crate::const_splat!(f32: 2.4187564849853515625E-4 * 2.0);
        let dp3f = crate::const_splat!(f32: 3.77476681023836135864E-8 * 2.0);
        let dp4f = crate::const_splat!(f32: 1.28164145962728071027E-12 * 2.0);

        if const { V::HAS_TRUE_FMA } {
            // dp1f + dp2f is exact in f32; three chained FMAs
            y.nmul_add(dp4f, y.nmul_add(dp3f, y.nmul_add(dp2f + dp1f, xa)))
        } else {
            (((xa - y * dp1f) - y * dp2f) - y * dp3f) - y * dp4f
        }
    };

    let mut x_lo = V::ZERO;

    // Payne-Hanek fallback for large arguments (non-PI only, Best+ precision)
    if const { P::POLICY.precision.gt(PrecisionPolicy::Average) && !PI }
        && (P::POLICY.avoid_branching || is_large.any())
    {
        let (x_ph, x_lo_ph, q_ph) = payne_hanek_reduction::<P, V>(&xa);

        x = is_large.select(x_ph, x);
        x_lo = x_lo_ph.zz(is_large); // zero out x_lo when not using Payne-Hanek
        q = is_large.select(q_ph, q);
    }

    (x, x_lo, q)
}

#[inline(always)]
fn sin_cos_f_internal<P: Policy, V: FloatVectorWithBits<Element = f32>, const PI: bool, const SINGLE: bool>(
    xx: V,
) -> (V, V) {
    if const { SINGLE && P::POLICY.precision.le(PrecisionPolicy::Worst) } {
        // Max error about 0.00092, avg error about 0.00053
        // https://stackoverflow.com/a/28050328/2083075
        // the actual instruction count isn't that much better,
        // but it avoids integer conversions and branches
        #[inline(always)] #[rustfmt::skip]
        fn inner<V: FloatVector<Element = f32>>(mut x: V) -> V {
            // rearrange for FMA, no chance of overflow since x is (-0.5, 0.5) here
            //x *= V::splat(16.0) * (x.abs() - V::splat(0.5));
            x *= x.abs().mul_sube(
                crate::const_splat!(f32: 16.0),
                crate::const_splat!(f32: 8.0),
            );

            // https://stackoverflow.com/questions/18662261/#comment138971102_28050328
            // increases average error but decreases max error
            let p = crate::const_splat!(f32: 0.22400815333595678); // original P = 0.225

            x.mul_adde(x.abs().mul_sube(p, p), x)
        }

        let xx = xx.flush_denormals::<P>();

        // scaling factor
        let m = if PI {
            V::HALF // (1/pi) / 2 * pi = 0.5
        } else {
            crate::const_splat!(f32: FRAC_1_PI / 2.0)
        };

        return if const { V::HAS_TRUE_FMA && V::ISA.has_instruction_level_parallelism() } {
            // if FMA is available, we can improve ILP by doing product with m in parallel
            (
                inner::<V>(xx.mul_sub(m, V::HALF) - (xx * m).floor()), // sine
                inner::<V>(xx.mul_sub(m, V::FRAC_1_4) - xx.mul_add(m, V::FRAC_1_4).floor()), // cosine
            )
        } else {
            let x = m * xx;

            (
                inner::<V>((x - V::HALF) - x.floor()),                     // sine
                inner::<V>((x - V::FRAC_1_4) - (x + V::FRAC_1_4).floor()), // cosine
            )
        };
    }

    let xa = xx.abs().flush_denormals::<P>();

    let (x, x_lo, q) = trig_range_reduction::<P, V, PI>(xa);

    // Taylor expansion of sin and cos, valid for -pi/4 <= x <= pi/4
    let x2 = x * x;
    let mut x0 = x;

    if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
        x0 += x_lo;
    }

    #[rustfmt::skip]
    let mut s = x2.poly_rev_p::<P, _>(&[
        -1.9515295891E-4,
        8.3321608736E-3,
        -1.6666654611E-1,
    ])
    .mul_adde(x2 * x, x0);

    #[rustfmt::skip]
    let mut c = x2.poly_rev_p::<P, _>(&[
        2.443315711809948E-5,
        -1.388731625493765E-3,
        4.166664568298827E-2,
    ])
    .mul_adde(x2 * x2, x2.nmul_adde(V::HALF, V::ONE));

    if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
        c = x.nmul_adde(x_lo, c);
    }

    let swap = (q & V::Bits::ONE).cmp_ne(V::Bits::ZERO);

    let sin1 = swap.select(c, s);
    let cos1 = swap.select(s, c);

    let signsin = V::from_bits(q.shli::<30>()) ^ xx;
    let signcos = V::from_bits((q + V::Bits::ONE).shri::<1>().shli::<31>());

    (sin1.mul_sign(signsin), cos1 ^ signcos)
}

#[inline(always)]
fn asin_f_internal<P: Policy, V: FloatVectorWithBits<Element = f32>, const ACOS: bool>(x: V) -> V {
    let xa = x.abs().flush_denormals::<P>();

    if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
        /* Based on http://www.pouet.net/topic.php?which=9132&page=2
         * 85% accurate (ULP 0)
         * Examined 2130706434 values of acos:
         *   15.2000597 avg ULP diff, 4492 max ULP, 4.51803e-05 max error // without "denormal crush"
         * Examined 2130706434 values of acos:
         *   15.2007108 avg ULP diff, 4492 max ULP, 4.51803e-05 max error // with "denormal crush"
         */
        let mut m = xa.min(V::ONE); // clamp

        let a0 = (V::ONE - m).sqrt();
        let a1 = m.poly_rev_p::<P, _>(&[-0.02164095, 0.077980478, -0.213300989, FRAC_PI_2]);

        if ACOS {
            if const { V::HAS_TRUE_FMA && V::ISA.has_instruction_level_parallelism() } {
                // if FMA is available we can at least exploit instruction-level parallelism
                return x.select_negative(a0.nmul_add(a1, V::PI), a0 * a1);
            }

            let a = a0 * a1;
            return x.select_negative(V::PI - a, a);
        } else {
            // Max error is 4.51133e-05 (ULPS are higher because we are consistently off by a little amount).
            return a0.nmul_adde(a1, V::FRAC_PI_2).copysign(x);
        }
    }

    let is_big = xa.cmp_gt(V::HALF);

    // TODO: Branch to avoid sqrt?
    let x1 = V::HALF * (V::ONE - xa);
    let x3 = is_big.select(x1, xa * xa);
    let x4 = is_big.select(x1.sqrt(), xa);

    #[rustfmt::skip]
    let z = x3.poly_rev_p::<P, _>(&[
        4.2163199048E-2,
        2.4181311049E-2,
        4.5470025998E-2,
        7.4953002686E-2,
        1.6666752422E-1,
    ])
    .mul_adde(x3 * x4, x4);

    let z1 = z + z;

    if ACOS {
        let z1 = x.select_negative(V::PI - z1, z1);
        let z2 = V::FRAC_PI_2 - z.mul_sign(x);

        is_big.select(z1, z2)
    } else {
        let z1 = V::FRAC_PI_2 - z1;

        is_big.select(z1, z).mul_sign(x)
    }
}

#[inline(always)]
fn pow2n_f<V: FloatVectorWithBits<Element = f32>>(n: V) -> V {
    let pow2_23: V = crate::const_splat!(f32: 8388608.0);
    let bias: V = crate::const_splat!(f32: 127.0);

    V::from_bits(V::Bits::from_bits(n + (bias + pow2_23)).shli::<23>())
}

/// Split 2^r into two multiplications so neither one leaves normal range
#[inline(always)]
fn pow2n_f_safe<V: FloatVectorWithBits<Element = f32>>(n: V) -> (V, V) {
    // Split n into two halves, each in [-126, 127]
    let half = n.scale(0.5).floor();
    let other = n - half;
    (pow2n_f(half), pow2n_f(other))
}

#[inline(always)]
fn exp_f_internal<P: Policy, V: FloatVectorWithBits<Element = f32>, const MODE: u8>(x0: V) -> V {
    if const { P::POLICY.precision.le(PrecisionPolicy::Average) } {
        if const { V::NATIVE_CAP.has(NativeCapability::EXP) && MODE == EXP_MODE_EXP } {
            return unsafe { x0.native_exp::<P>() };
        }

        if const { V::NATIVE_CAP.has(NativeCapability::EXP2) && MODE == EXP_MODE_POW2 } {
            return unsafe { x0.native_exp2::<P>() };
        }
    }

    let x0 = x0.flush_denormals::<P>();

    let mut x = x0;
    let mut r;

    let mut z = if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
        // Compute t such that b^x = 2^t
        let t = match MODE {
            EXP_MODE_EXP | EXP_MODE_EXPH | EXP_MODE_EXPM1 => x.scale(FloatConsts::LOG2_E),
            EXP_MODE_POW10 | EXP_MODE_POW10M1 => x.scale(FloatConsts::LOG2_10),
            EXP_MODE_POW2 | EXP_MODE_POW2M1 => x,
            _ => unreachable!("Invalid MODE for exp_f_internal"),
        };

        let fi = t.floor();
        let f = t - fi;

        // if the exponent exceeds this method's limitations, then it's far outside of the valid range for exp
        let i: V::SignedBits = fi.fast_cast();

        // polynomial approximation of 2^f in [0, 1) either using a degree-7 or degree-3 polynomial
        // these are noteworthy because degree-7 is _barely_ more expensive than degree-3 if using Estrin's scheme
        // and instruction-level parallelism is a thing.
        let cf = if const { P::POLICY.precision.gt(PrecisionPolicy::Worst) } {
            // max. rel. error <= ~7.55e-11 on [0,1) via Sollya,
            // which is perfect for f32, but Medium precision
            // has worse range reduction
            f.poly_rev_p::<P, _>(&[
                0.000021428975742310286, // c7 - 0x37b3c260
                0.000143863057019189000, // c6 - 0x3916d9f2
                0.001341646537184715271, // c5 - 0x3aafda30
                0.009614554233849048615, // c4 - 0x3c1d865d
                0.055504892021417617798, // c3 - 0x3d635919
                0.240226432681083679199, // c2 - 0x3e75fdeb
                0.693147182464599609375, // c1 - 0x3f317218 (≈ ln2)
                1.0,                     // c0 - exact
            ])
        } else {
            // https://stackoverflow.com/a/10792321 with a better 2^f fit
            // max. rel. error <= 1.73e-3 on [-87,88]
            f.poly_rev_p::<P, _>(&[0.0781455737, 0.226173572, 0.695556856, 1.0])
        };

        // scale 2^f by 2^i
        let ci = V::SignedBits::from_bits(cf) + (i << 23);

        let z = V::from_bits(ci);

        match MODE {
            EXP_MODE_EXPH => z.scale(0.5),
            EXP_MODE_EXPM1 | EXP_MODE_POW2M1 | EXP_MODE_POW10M1 => z - V::ONE,
            EXP_MODE_EXP | EXP_MODE_POW2 | EXP_MODE_POW10 => z,
            _ => unreachable!("Invalid MODE for exp_f_internal"),
        }
    } else {
        match MODE {
            EXP_MODE_POW2 | EXP_MODE_POW2M1 => {
                r = x0.round();

                x -= r;
                x *= V::LN_2;
            }
            EXP_MODE_POW10 | EXP_MODE_POW10M1 => {
                let log10_2_hi: V = crate::const_splat!(f32: -0.301025391); // log10(2) in two parts
                let log10_2_lo: V = crate::const_splat!(f32: -4.60503907E-6);

                // TODO: Combine these constants and use .scale()
                r = (x0 * crate::const_splat!(f32: LN_10 * LOG2_E)).round();

                x = r.mul_adde(log10_2_hi, x); // x -= r * log10_2_hi;
                x = r.mul_adde(log10_2_lo, x); // x -= r * log10_2_lo;
                x *= V::LN_10;
            }
            EXP_MODE_EXP | EXP_MODE_EXPM1 | EXP_MODE_EXPH => {
                let ln2f_hi: V = crate::const_splat!(f32: -0.693359375);
                let ln2f_lo: V = crate::const_splat!(f32: 2.12194440e-4);

                r = x0.scale(FloatConsts::LOG2_E).round();

                x = r.mul_adde(ln2f_hi, x); // x -= r * ln2f_hi;
                x = r.mul_adde(ln2f_lo, x); // x -= r * ln2f_lo;

                if const { MODE == EXP_MODE_EXPH } {
                    r -= V::ONE;
                }
            }
            _ => unreachable!("Invalid MODE for exp_f_internal"),
        }

        let z = x
            .poly_rev_p::<P, _>(&[1.0 / 5040.0, 1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 1.0 / 2.0])
            .mul_adde(x * x, x);

        if const { P::POLICY.precision.le(PrecisionPolicy::Average) } {
            let n2 = pow2n_f::<V>(r);

            match MODE {
                EXP_MODE_EXPM1 | EXP_MODE_POW2M1 | EXP_MODE_POW10M1 => z.mul_adde(n2, n2 - V::ONE),
                _ => z.mul_adde(n2, n2), // (z + 1.0f) * n2
            }
        } else {
            let (n2a, n2b) = pow2n_f_safe::<V>(r);

            match MODE {
                EXP_MODE_EXPM1 | EXP_MODE_POW2M1 | EXP_MODE_POW10M1 => {
                    z.mul_adde(n2a, n2a - V::ONE).mul_adde(n2b, n2b - V::ONE)
                }
                _ => z.mul_adde(n2a, n2a) * n2b, // (z + 1) * n2a * n2b
            }
        }
    };

    if const { P::POLICY.check_overflow } {
        let mut in_range = x0.is_finite();

        if const { P::POLICY.precision.gt(PrecisionPolicy::Average) } {
            #[rustfmt::skip]
            let (min_x, max_x) = const { match MODE {
                EXP_MODE_EXP => (-103.97, 88.72),  // (ln(2^-150), ln(FLT_MAX))
                EXP_MODE_EXPM1 => (-87.0, 88.72),  // ln(FLT_MAX)
                EXP_MODE_EXPH => (-103.97, 89.42), // ln(2 * FLT_MAX)
                EXP_MODE_POW2 => (-150.0, 128.0),  // (2^-150 rounds to 0, log2(FLT_MAX))
                EXP_MODE_POW2M1 => (-150.0, 128.0), // (2^x - 1 -> -1 below, log2(FLT_MAX))
                EXP_MODE_POW10 => (-45.15, 38.53), // (log10(2^-150), log10(FLT_MAX))
                EXP_MODE_POW10M1 => (-45.15, 38.53), // (10^x - 1 -> -1 below, log10(FLT_MAX))

                _ => panic!("Invalid MODE for exp_f_internal"), // unreachable!() isn't const apparently
            }};

            in_range &= x0.cmp_ge(V::splat(min_x)) & x0.cmp_le(V::splat(max_x));
        } else {
            #[rustfmt::skip]
            let max_x = const { match MODE {
                EXP_MODE_EXP => 87.3,
                EXP_MODE_POW2 | EXP_MODE_POW2M1 => 126.0,
                EXP_MODE_POW10 | EXP_MODE_POW10M1 => 37.9,
                EXP_MODE_EXPH | EXP_MODE_EXPM1 => 89.0,

                _ => panic!("Invalid MODE for exp_f_internal"),
            }};

            in_range &= x0.abs().cmp_le(V::splat(max_x)); // symmetric limits for lesser precisions
        }

        // TODO: Investigate performance of this branch
        // if !P::POLICY.avoid_branching && crate::likely(in_range.all()) {
        //     return z;
        // }

        #[rustfmt::skip]
        let underflow_value = const { match MODE {
            EXP_MODE_EXPM1 | EXP_MODE_POW2M1 | EXP_MODE_POW10M1 => V::NEG_ONE,
            _ => V::ZERO,
        } };

        r = x0.select_negative(underflow_value, V::INFINITY);
        z = in_range.select(z, r);
        z = x0.is_nan().select(x0, z);
    }

    z
}

#[inline(always)]
fn fraction2<V: FloatVectorWithBits<Element = f32>>(x: V) -> V {
    // set exponent to 0 + bias
    let b = crate::const_splat!(f32: f32::from_bits(0x007FFFFF));
    let c = crate::const_splat!(f32: f32::from_bits(0x3F000000));

    //(x & b) | c
    V::ternlog::<{ crate::ternlog_imm!((A & B) | C) }>(x, b, c)
}

#[inline(always)]
fn exponent<V: FloatVectorWithBits<Element = f32>>(x: V) -> V::SignedBits {
    // shift out sign, extract exp, subtract bias
    V::SignedBits::from_bits((V::Bits::from_bits(x).shli::<1>()).shri::<24>()) - V::SignedBits::splat(0x7F)
}

#[inline(always)]
fn ln_2_internal<P: Policy, V: FloatVectorWithBits<Element = f32>>(x: V) -> V {
    if const { P::POLICY.precision.le(PrecisionPolicy::Average) && V::NATIVE_CAP.has(NativeCapability::LOG2) } {
        return unsafe { x.native_log2::<P>() };
    }

    if const { P::POLICY.precision.eq(PrecisionPolicy::Worst) } {
        // // https://github.com/nadavrot/fast_log/blob/83bd112c330976c291300eaa214e668f809367ab/src/log_approx.cc#L47
        // return fraction2::<V>(x).poly_p::<P, _>(&[-3.21430967, 6.30371424, -4.42852392, 1.33755322])
        //     + (exponent::<V>(x) + V::SignedBits::ONE).cast();

        // https://github.com/romeric/fastapprox/blob/ccc534400ec3e0f67de4eafb53377334962d9db6/fastapprox/src/fastonebigheader.h#L384
        // between 1e-4 and 1000, avg error: 0.00536, max error 0.0573 at 31.999878
        return V::cast_from(V::SignedBits::from_bits(x)).mul_sube(
            crate::const_splat!(f32: 1.1920928955078125e-7),
            crate::const_splat!(f32: 126.94269504),
        );
    }

    ln_f_internal::<P, V, false>(x).scale(FloatConsts::LOG2_E)
}

#[inline(always)]
fn ln_10_internal<P: Policy, V: FloatVectorWithBits<Element = f32>>(x: V) -> V {
    if const { P::POLICY.precision.le(PrecisionPolicy::Average) && V::NATIVE_CAP.has(NativeCapability::LOG2) } {
        return unsafe { x.native_log2::<P>().scale(FloatConsts::LOG10_2) };
    }

    if const { P::POLICY.precision.eq(PrecisionPolicy::Worst) } {
        // ln(x) * LOG10_E
        // between 1e-4 and 1000, avg error: 0.00212, max error 0.0173 at 31.999878
        return V::cast_from(V::SignedBits::from_bits(x)).mul_sube(
            crate::const_splat!(f32: 3.5885571887588505e-8),
            crate::const_splat!(f32: 38.213558906),
        );
    }

    ln_f_internal::<P, V, false>(x).scale(FloatConsts::LOG10_E)
}

#[inline(always)]
fn ln_f_internal<P: Policy, V: FloatVectorWithBits<Element = f32>, const P1: bool>(x0: V) -> V {
    // TODO: How to handle P1?
    if const { P::POLICY.precision.le(PrecisionPolicy::Average) && V::NATIVE_CAP.has(NativeCapability::LN) && !P1 } {
        return unsafe { x0.native_ln::<P>() };
    }

    if const { P::POLICY.precision.eq(PrecisionPolicy::Worst) } {
        let x1 = if P1 { x0 + V::ONE } else { x0 };

        // https://github.com/romeric/fastapprox/blob/ccc534400ec3e0f67de4eafb53377334962d9db6/fastapprox/src/fastonebigheader.h#L393
        // between 1e-4 and 1000, avg error: 0.00536, max error 0.0397 at 3.9999847
        return V::cast_from(V::SignedBits::from_bits(x1)).mul_sube(
            crate::const_splat!(f32: 8.2629582881927490e-8),
            crate::const_splat!(f32: 87.989971088),
        );
    }

    if const { P::POLICY.precision.eq(PrecisionPolicy::Medium) } {
        // https://stackoverflow.com/a/39822314/2083075
        // natural log on [0x1.f7a5ecp-127, 0x1.fffffep127]. Maximum relative error 9.4529e-5

        let a = V::SignedBits::from_bits(x0);
        let e = (a - V::SignedBits::splat(0x3f2aaaab)) & V::SignedBits::splat(0xff800000u32 as i32);
        let i = V::cast_from(e) * crate::const_splat!(f32: 1.19209290e-7);
        let mut f = V::from_bits(a - e);

        if !P1 {
            f -= V::ONE;
        }

        let s = f * f;

        /* Compute log1p(f) for f in [-1/3, 1/3] */
        let r = f.mul_adde(
            crate::const_splat!(f32: 0.230836749),
            crate::const_splat!(f32: -0.279208571),
        ); // 0x1.d8c0f0p-3, -0x1.1de8dap-2
        let t = f.mul_adde(
            crate::const_splat!(f32: 0.331826031),
            crate::const_splat!(f32: -0.498910338),
        ); // 0x1.53ca34p-2, -0x1.fee25ap-2
        let r = r.mul_adde(s, t).mul_adde(s, f);
        let r = i.mul_adde(crate::const_splat!(f32: 0.693147182), r); // 0x1.62e430p-1 // log(2)

        return r;
    }

    let x0 = x0.flush_denormals::<P>();

    let ln2f_hi = crate::const_splat!(f32: 0.693359375);
    let ln2f_lo = crate::const_splat!(f32: -2.12194440E-4);

    let x1 = if P1 { x0 + V::ONE } else { x0 };

    let mut x = fraction2::<V>(x1);
    let mut e = exponent::<V>(x1);

    let blend = x.cmp_gt(crate::const_splat!(f32: SQRT_2 * 0.5));

    x = x.add_c(!blend, x);
    e = e.add_c(blend.cast(), V::SignedBits::ONE);

    let fe: V = e.cast();

    let xp1 = x - V::ONE;

    x = if P1 {
        // log(x+1). Avoid loss of precision when adding 1 and later subtracting 1 if exponent = 0
        e.cmp_eq(V::SignedBits::ZERO).select(x0, xp1)
    } else {
        xp1 // log(x). Expand around 1.0
    };

    let x2 = x * x;
    let mut res = x.poly_rev_p::<P, _>(&[
        7.0376836292E-2,
        -1.1514610310E-1,
        1.1676998740E-1,
        -1.2420140846E-1,
        1.4249322787E-1,
        -1.6668057665E-1,
        2.0000714765E-1,
        -2.4999993993E-1,
        3.3333331174E-1,
        0.0, // multiply all by x
    ]);

    res = fe.mul_adde(ln2f_lo, res.mul_adde(x2, x2.nmul_adde(V::HALF, x)));
    res = fe.mul_adde(ln2f_hi, res);

    if const { !P::POLICY.check_overflow } {
        return res;
    }

    let overflow = !x1.is_finite();
    let underflow = x1.cmp_lt(crate::const_splat!(f32: 1.17549435e-38));

    if const { !P::POLICY.avoid_branching } && crate::likely((overflow | underflow).none()) {
        return res;
    }

    res = underflow.select(V::NAN, res); // x1 < 0 gives NAN
    res = x1.is_zero_or_subnormal().select(V::NEG_INFINITY, res); // x1 == 0 gives -INF
    res = overflow.select(x1, res); // INF or NAN goes through
    res = (x1.is_infinite() & x1.is_negative()).select(V::NAN, res); // -INF gives NAN

    res
}
