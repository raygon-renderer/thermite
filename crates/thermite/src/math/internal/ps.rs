use crate::{
    divider::Divider,
    math::{
        consts::FloatConsts,
        policy::policies::{ExtraPrecision, MediumPrecision},
    },
};
use core::f32::consts::{FRAC_1_PI, FRAC_PI_2, LN_10, LOG2_E, SQRT_2};

use super::*;

impl<R> MathInternal<f32> for R
where
    R: FloatRegister<Element = f32>,
{
    #[inline(always)]
    fn sincos<P: Policy>(xx: Vf<Self>) -> (Vf<Self>, Vf<Self>) {
        if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
            // Max error about 0.00092
            // https://stackoverflow.com/a/28050328/2083075
            #[inline(always)]
            fn fast_sin_cos<R: MathInternal<f32>, const SINE: bool>(mut x: Vf<R>) -> Vf<R> {
                // encourage instruction-level parallelism
                if SINE {
                    x = (x - Vf::HALF) - x.floor();
                } else {
                    let quarter = Vf::splat(0.25);

                    x = (x - quarter) - (x + quarter).floor();
                }

                // rearrange for FMA, no chance of overflow since x is (-0.5, 0.5) here
                //x *= Vf::splat(16.0) * (x.abs() - Vf::splat(0.5));
                x *= x.abs().mul_sube(Vf::splat(16.0), Vf::splat(8.0));

                // https://stackoverflow.com/questions/18662261/#comment138971102_28050328
                // increases average error but decreases max error
                let p = Vf::splat(0.22400815333595678); // original P = 0.225

                x.mul_adde(x.abs().mul_sube(p, p), x)
            }

            let x = xx * Vf::splat(FRAC_1_PI / 2.0);

            let sine = fast_sin_cos::<R, true>(x);
            let cosine = fast_sin_cos::<R, false>(x);

            return (sine, cosine);
        }

        let xa = xx.abs();

        let y = (xa * Vf::FRAC_2_PI).round();
        let q: Vu<Self> = Vs::<Self>::fast_from(y).into_bits();

        let dp1f = const { Vector::splat_const(0.78515625 * 2.0) };
        let dp2f = const { Vector::splat_const(2.4187564849853515625E-4 * 2.0) };
        let dp3f = const { Vector::splat_const(3.77489497744594108E-8 * 2.0) };

        // Reduce by extended precision modular arithmetic
        // x = ((xa - y * DP1F) - y * DP2F) - y * DP3F;
        let x = y.nmul_adde(dp3f, y.nmul_adde(dp2f, y.nmul_adde(dp1f, xa)));

        // Taylor expansion of sin and cos, valid for -pi/4 <= x <= pi/4
        let x2 = x * x;

        #[rustfmt::skip]
        let mut s = x2.poly_p::<P, _>(&[
            -1.6666654611E-1,
            8.3321608736E-3,
            -1.9515295891E-4,
        ])
        .mul_adde(x2 * x, x);

        #[rustfmt::skip]
        let mut c = x2.poly_p::<P, _>(&[
            4.166664568298827E-2,
            -1.388731625493765E-3,
            2.443315711809948E-5,
        ])
        .mul_adde(x2 * x2, x2.nmul_adde(Vf::HALF, Vf::ONE));

        let swap = (q & Vu::<Self>::ONE).cmp_ne(Vu::<Self>::ZERO);

        let sin1 = swap.select(c, s);
        let cos1 = swap.select(s, c);

        let signsin = Vf::from_bits(q.shli::<30>()) ^ xx;
        let signcos = Vf::from_bits((q + Vu::<Self>::ONE).shri::<1>().shli::<31>());

        (sin1.mul_sign(signsin), cos1 ^ signcos)
    }

    #[inline(always)]
    fn sinh<P: Policy>(x0: Vf<Self>) -> Vf<Self> {
        let x = x0.abs();

        let x_small = x.cmp_lt(Vf::ONE);

        let mut y2 = Vf::EMPTY;

        // if not all are small, use exponential functions
        if P::POLICY.avoid_branching || !x_small.all() {
            y2 = Self::exph::<P>(x);
            y2 -= Vf::splat(0.25) / y2;

            if const { P::POLICY.avoid_precision_branches() } {
                return y2.mul_sign(x0);
            }
        }

        // if all are small, use a polynomial approximation
        if P::POLICY.avoid_branching || x_small.any() {
            let x2 = x * x;

            let y1 = x2
                .poly_p::<P, _>(&[1.66667160211E-1, 8.33028376239E-3, 2.03721912945E-4])
                .mul_adde(x2 * x, x);

            y2 = x_small.select(y1, y2);
        }

        y2.mul_sign(x0)
    }

    #[inline(always)]
    fn cosh<P: Policy>(x0: Vf<Self>) -> Vf<Self> {
        let y = Self::exph::<P>(x0.abs());
        y + Vf::splat(0.25) / y
    }

    #[inline(always)]
    #[rustfmt::skip]
    fn tanh<P: Policy>(x0: Vf<Self>) -> Vf<Self> {
        let one = Vf::ONE;

        let x = x0.abs();
        let x_small = x.cmp_lt(Vf::splat(0.625));

        let mut y2 = Vf::EMPTY;

        // if not all are small
        if P::POLICY.avoid_branching || !x_small.all() {
            y2 = Self::exp::<P>(x + x);
            // originally (1 - 2/(y2 + 1)), but doing it this way avoids
            // loading 2.0 and encourages slight instruction-level parallelism
            y2 = (y2 - one) / (y2 + one);

            if P::POLICY.check_overflow {
                y2 = x.cmp_gt(Vf::splat(44.4)).select(one, y2);
            }

            if P::POLICY.avoid_precision_branches() {
                return y2.mul_sign(x0);
            }
        }

        // if any are small
        if P::POLICY.avoid_branching || x_small.any() {
            let x2 = x * x;

            let y1 = x2.poly_p::<P, _>(&[
                -3.33332819422E-1,
                1.33314422036E-1,
                -5.37397155531E-2,
                2.06390887954E-2,
                -5.70498872745E-3,
            ]).mul_adde(x2 * x, x);

            y2 = x_small.select(y1, y2);
        }

        y2.mul_sign(x0)
    }

    #[inline(always)]
    fn asin<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        asin_f_internal::<P, Self, false>(x)
    }

    #[inline(always)]
    fn acos<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        asin_f_internal::<P, Self, true>(x)
    }

    #[inline(always)]
    fn atan<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        let t = x.abs();

        if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
            /* http://mathforum.org/library/drmath/view/62672.html
             * Examined 4278190080 values of atan:
             *   2.36864877 avg ULP diff, 302 max ULP, 6.55651e-06 max error      // (with  denormals)
             * Examined 4278190080 values of atan:
             *   171160502 avg ULP diff, 855638016 max ULP, 6.55651e-06 max error // (crush denormals)
             */
            let a = t;

            let gt1 = a.cmp_gt(Vf::ONE);

            let mut s = gt1.select(a.reciprocal_p::<ExtraPrecision<P>>(), a);

            s = Vf::ONE - (Vf::ONE - s); // crush denormals

            let t = s * s;

            // place the s * 0.43157974 in the FMA to encourage instruction-level parallelism
            let r = t.mul_adde(s * Vf::splat(0.43157974), Vf::ONE)
                / t.mul_add(Vf::splat(0.05831938), Vf::splat(0.76443945))
                    .mul_add(t, Vf::ONE);

            let r = gt1.select(Vf::FRAC_PI_2 - r, r);

            return r.copysign(x);
        }

        let not_small = t.cmp_ge(Vf::splat(SQRT_2 - 1.0)); // t >= tan  pi/8
        let not_big = t.cmp_le(Vf::splat(SQRT_2 + 1.0)); // t <= tan 3pi/8

        let s = not_big.select(Vf::FRAC_PI_4, Vf::FRAC_PI_2) & not_small.value(); // select(not_small, s, 0.0);

        // small:  z = t / 1.0;
        // medium: z = (t-1.0) / (t+1.0);
        // big:    z = -1.0 / t;

        // this trick avoids having to place a zero in any register
        let a = (not_big.value() & t) + (not_small.value() & Vf::NEG_ONE);
        let b = (not_big.value() & Vf::ONE) + (not_small.value() & t);

        let z = a / b;
        let z2 = z * z;

        z2.poly_p::<P, _>(&[-3.33329491539E-1, 1.99777106478E-1, -1.38776856032E-1, 8.05374449538E-2])
            .mul_adde(z2 * z, z + s)
            .mul_sign(x)
    }

    #[inline(always)]
    fn atan2<P: Policy>(y: Vf<Self>, x: Vf<Self>) -> Vf<Self> {
        let neg_one = Vf::NEG_ONE;
        let zero = Vf::ZERO;

        let x1 = x.abs();
        let y1 = y.abs();

        let swap_xy = y1.cmp_gt(x1);

        if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
            let (a, b) = (x1, y1);

            let n = swap_xy.select(b, a);
            let d = swap_xy.select(a, b);

            let mut k = n / d;

            if P::POLICY.check_overflow {
                let b_eq_zero = b.cmp_eq(Vf::ZERO);
                let ab_eq = a.cmp_eq(b);

                k = ab_eq.select(Vf::ONE, k);
                k = b_eq_zero.select(Vf::ZERO, k);
            }

            let s = Vf::ONE - (Vf::ONE - k); // crush denormals

            let t = s * s;

            let mut r = t.mul_adde(s * Vf::splat(0.43157974), Vf::ONE)
                / t.mul_add(Vf::splat(0.05831938), Vf::splat(0.76443945))
                    .mul_add(t, Vf::ONE);

            r = swap_xy.select(Vf::FRAC_PI_2 - r, r);
            r = x.select_negative(Vf::PI - r, r);

            return r.copysign(y);
        }

        let mut x2 = swap_xy.select(y1, x1);
        let mut y2 = swap_xy.select(x1, y1);

        if P::POLICY.check_overflow {
            let both_infinite = (x.is_infinite() & y.is_infinite());

            //if crate::unlikely(both_infinite.any())
            x2 = both_infinite.select(x2 & neg_one, x2); // get 1.0 with the sign of x
            y2 = both_infinite.select(y2 & neg_one, y2); // get 1.0 with the sign of y
        }

        // x = y = 0 will produce NAN. No problem, fixed below
        let t = y2 / x2;

        // small:  z = t / 1.0;
        // medium: z = (t-1.0) / (t+1.0);
        let not_small = t.cmp_ge(Vf::splat(SQRT_2 - 1.0));

        let a = t + (not_small.value() & neg_one);
        let b = Vf::ONE + (not_small.value() & t);

        let s = not_small.value() & Vf::FRAC_PI_4;

        let z = a / b;
        let z2 = z * z;

        let mut re = z2
            .poly_p::<P, _>(&[-3.33329491539E-1, 1.99777106478E-1, -1.38776856032E-1, 8.05374449538E-2])
            .mul_adde(z2 * z, z + s);

        re = swap_xy.select(Vf::FRAC_PI_2 - re, re);
        re = (x | y).cmp_eq(zero).select(zero, re); // atan2(0,+0) = 0 by convention
        re = x.select_negative(Vf::PI - re, re); // also for x = -0.

        re
    }

    #[inline(always)]
    fn asinh<P: Policy>(x0: Vf<Self>) -> Vf<Self> {
        let x = x0.abs();
        let x2 = x0 * x0;

        let x_small = x.cmp_le(Vf::splat(0.51));

        let mut y1 = Vf::EMPTY;
        let mut y2 = Vf::EMPTY;

        if P::POLICY.avoid_branching || !x_small.all() {
            y2 = ((x2 + Vf::ONE).sqrt() + x).ln_p::<P>();

            if P::POLICY.check_overflow {
                let x_huge = x.cmp_gt(Vf::splat(1e10));

                if P::POLICY.avoid_precision_branches() || crate::unlikely(x_huge.any()) {
                    y2 = x_huge.select(x.ln_p::<P>() + Vf::LN_2, y2);
                }
            }

            if P::POLICY.avoid_precision_branches() {
                return y2.mul_sign(x0);
            }
        }

        if P::POLICY.avoid_branching || x_small.any() {
            y1 = x2
                .poly_p::<P, _>(&[-1.6666288134E-1, 7.4847586088E-2, -4.2699340972E-2, 2.0122003309E-2])
                .mul_adde(x2 * x, x);

            y2 = x_small.select(y1, y2);
        }

        y2.mul_sign(x0)
    }

    #[inline(always)]
    fn acosh<P: Policy>(x0: Vf<Self>) -> Vf<Self> {
        let one = Vf::ONE;

        let x1 = x0 - one;

        let x_small = x1.cmp_lt(Vf::splat(0.49)); // use Pade approximation if abs(x-1) < 0.5

        let mut y1 = Vf::EMPTY;
        let mut y2 = Vf::EMPTY;

        // if not all are small
        if P::POLICY.avoid_branching || !x_small.all() {
            y2 = (x0.mul_sube(x0, one).sqrt() + x0).ln_p::<P>();

            if P::POLICY.check_overflow {
                let x_huge = x1.cmp_gt(Vf::splat(1e10));

                if P::POLICY.avoid_precision_branches() || crate::unlikely(x_huge.any()) {
                    y2 = x_huge.select(x0.ln_p::<P>() + Vf::LN_2, y2);
                }
            }

            if P::POLICY.avoid_precision_branches() {
                return y2;
            }
        }

        // if any are small
        if P::POLICY.avoid_branching || x_small.any() {
            y1 = x1.sqrt()
                * x1.poly_p::<P, _>(&[
                    1.4142135263E0,
                    -1.1784741703E-1,
                    2.6454905019E-2,
                    -7.5272886713E-3,
                    1.7596881071E-3,
                ]);

            if P::POLICY.check_overflow {
                // result is NaN if less-than 1
                y1 = x0.cmp_lt(one).select(Vf::NAN, y1);
            }

            y2 = x_small.select(y1, y2);
        }

        y2
    }

    #[inline(always)]
    fn atanh<P: Policy>(x0: Vf<Self>) -> Vf<Self> {
        let x = x0.abs();

        let x_small = x.cmp_lt(Vf::splat(0.5));

        let mut y1 = Vf::EMPTY;
        let mut y2 = Vf::EMPTY;

        if P::POLICY.avoid_branching || !x_small.all() {
            let one = Vf::ONE;

            y2 = ((one + x) / (one - x)).ln_p::<P>() * Vf::splat(0.5);

            if P::POLICY.check_overflow {
                let y3 = x.cmp_eq(one).select(Vf::INFINITY, Vf::NAN);
                y2 = x.cmp_ge(one).select(y3, y2);
            }

            if P::POLICY.avoid_precision_branches() {
                return y2.mul_sign(x0);
            }
        }

        if P::POLICY.avoid_branching || x_small.any() {
            let x2 = x * x;

            y1 = x2
                .poly_p::<P, _>(&[
                    3.33337300303E-1,
                    1.99782164500E-1,
                    1.46691431730E-1,
                    8.24370301058E-2,
                    1.81740078349E-1,
                ])
                .mul_adde(x2 * x, x);

            y2 = x_small.select(y1, y2);
        }

        y2.mul_sign(x0)
    }

    #[inline(always)]
    fn exp<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        exp_f_internal::<P, Self, EXP_MODE_EXP>(x)
    }

    #[inline(always)]
    fn exph<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        exp_f_internal::<P, Self, EXP_MODE_EXPH>(x)
    }

    #[inline(always)]
    fn exp2<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        exp_f_internal::<P, Self, EXP_MODE_POW2>(x)
    }

    #[inline(always)]
    fn exp10<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        exp_f_internal::<P, Self, EXP_MODE_POW10>(x)
    }

    #[inline(always)]
    fn exp_m1<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        exp_f_internal::<P, Self, EXP_MODE_EXPM1>(x)
    }

    #[inline(always)]
    fn powf<P: Policy>(x0: Vf<Self>, y: Vf<Self>) -> Vf<Self> {
        if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
            // the "Worst" log2 precision is _terrible_, so just use medium
            // to give anything reasonable back
            return (x0.log2_p::<MediumPrecision<P>>() * y).exp2_p::<P>();
        }

        // define constants
        let ln2f_hi = Vf::<R>::splat(0.693359375); // log(2), split in two for extended precision
        let ln2f_lo = Vf::<R>::splat(-2.12194440e-4);
        let log2e = Vf::LOG2_E;
        let ln2 = Vf::LN_2;

        let zero = Vf::ZERO;
        let one = Vf::ONE;
        let half = Vf::HALF;

        let x1 = x0.abs();

        let mut x = fraction2::<R>(x1);

        let blend = x.cmp_gt(Vf::<R>::splat(SQRT_2 * 0.5));

        // reduce range of x = +/- sqrt(2)/2
        x += blend.andnot(x); // !blend.value() & x;
        x -= one;

        // Taylor expansion, high precision
        let x2 = x * x;

        // logarithm expansion
        let mut lg1 = x.poly_p::<P, _>(&[
            3.3333331174E-1,
            -2.4999993993E-1,
            2.0000714765E-1,
            -1.6668057665E-1,
            1.4249322787E-1,
            -1.2420140846E-1,
            1.1676998740E-1,
            -1.1514610310E-1,
            7.0376836292E-2,
        ]);

        lg1 *= x2 * x;

        let ef = Vf::from(exponent::<R>(x1)) + (blend.value() & one);

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
        let x4 = x2 * x2;

        // Taylor expansion of exp
        let z = x
            .poly_p::<P, _>(&[1.0 / 2.0, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0, 1.0 / 720.0, 1.0 / 5040.0])
            .mul_adde(x * x, x + one);

        // contributions to exponent
        let ee = e1 + e2 + e3;
        let ei: Vs<R> = ee.fast_cast();

        // biased exponent of result:
        let ej = ei + (Vs::<R>::from_bits(z.abs()) >> 23);

        // add exponent by signed integer addition
        let mut z = Vf::<R>::from_bits(Vs::<R>::from_bits(z) + (ei << 23));

        if !P::POLICY.check_overflow {
            return z;
        }

        // check exponent for overflow and underflow
        let overflow = ej.cmp_ge(Vs::<R>::splat(0x0FF)).cast() | ee.cmp_gt(Vf::<R>::splat(300.0));
        let underflow = ej.cmp_le(Vs::<R>::splat(0x000)).cast() | ee.cmp_lt(Vf::<R>::splat(-300.0));

        // check for special cases
        let xfinite = x0.is_finite();
        let yfinite = y.is_finite();
        let efinite = ee.is_finite();

        let xzero = x0.is_zero_or_subnormal();
        let xsign = x0.is_negative();

        z = underflow.select(zero, z);
        z = overflow.select(Vf::INFINITY, z);

        let yzero = y.cmp_eq(zero);
        let yneg = y.cmp_lt(zero);

        // pow_case_x0
        z = xzero.select(yneg.select(Vf::INFINITY, yzero.select(one, zero)), z);

        let mut yodd = zero;

        if xsign.any() {
            let yint = y.cmp_eq(y.round());
            yodd = y << 31;

            let z0 = x0.cmp_eq(zero).select(z, Vf::NAN);
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
                .select(one, (x1.cmp_gt(one) ^ y.is_negative()).select(Vf::INFINITY, zero)),
        );

        // handle x infinite
        let z1 = xfinite.select(
            z1,
            yzero.select(
                one,
                yneg.select(
                    yodd & z,               // 0.0 with the sign of z from above
                    x0.abs() | (x0 & yodd), // get sign of x0 only if y is odd integer
                ),
            ),
        );

        // Always propagate nan:
        // Deliberately differing from the IEEE-754 standard which has pow(0,nan)=1, and pow(1,nan)=1
        (x0.is_nan() | y.is_nan()).select(x0 + y, z1)
    }

    #[inline(always)]
    fn cbrt<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        let b1 = Vu::<Self>::splat(709958130); // B1 = (127-127.0/3-0.03306235651)*2**23
        let b2 = Vu::<Self>::splat(642849266); // B2 = (127-127.0/3-24/3-0.03306235651)*2**23
        let m = Vu::<Self>::splat(0x7fffffff); // u32::MAX >> 1

        let x1p24 = x * Vf::splat(f32::from_bits(0x4b800000)); // 0x1p24f === 2 ^ 24

        let hx0 = x.into_bits() & m;

        let x_small = hx0.cmp_lt(Vu::<Self>::splat(0x00800000));

        let xs = x_small.select(x1p24, x);
        let b = x_small.select(b2, b1);

        let mut ui = xs.into_bits();
        let mut hx = ui & m;

        hx = hx / Divider::u32(3) + b;

        ui &= Vu::<Self>::splat(0x80000000);
        ui |= hx;

        let mut t = Vf::<Self>::from_bits(ui);

        if const { P::POLICY.precision.ge(PrecisionPolicy::Best) || !Self::HAS_TRUE_FMA } {
            // let mut td = t.cast::<Vf64<S>>();
            // let xd = x.cast::<Vf64<S>>();

            // // First iteration accurate to 16 bits, second iteration to 47 bits.
            // for _ in 0..2 {
            //     let r = td * td * td;
            //     let rxd = xd + r;
            //     td *= (xd + rxd) / (r + rxd);
            // }

            // t = <Vf32<S> as SimdFromCast<S, Vf64<S>>>::from_cast(td);

            todo!()
        } else {
            let two = Vf::TWO;

            // couple iterations of Newton's method
            // This isn't perfect, as it's only limited to single-precision,
            // but the fused multiply-adds helps
            for _ in 0..2 {
                let t3 = t * t * t;
                t *= two.mul_add(x, t3) / two.mul_add(t3, x); // try to use extended precision where possible
            }
        }

        if !P::POLICY.check_overflow {
            return x.cmp_eq(Vf::ZERO).select(x, t);
        }

        // cbrt(NaN,INF,+-0) is itself
        (hx0.cmp_gt(Vu::<Self>::splat(0x7f800000)) | hx0.cmp_eq(Vu::<Self>::ZERO)).select(x, t)
    }

    #[inline(always)]
    fn ln<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        ln_f_internal::<P, Self, false>(x)
    }

    #[inline(always)]
    fn ln1p<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        ln_f_internal::<P, Self, true>(x)
    }

    #[inline(always)]
    fn log2<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        ln_2_internal::<P, Self>(x)
    }

    #[inline(always)]
    fn log10<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        ln_10_internal::<P, Self>(x)
    }

    #[inline(always)]
    fn ln1m_expnx<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
            return Self::ln1m_expnx_ext::<P>(x, x.ln_p::<P>());
        }

        (Vf::ONE - (-x).exp_p::<P>()).ln_p::<P>()
    }

    #[inline(always)]
    fn ln1m_expnx_ext<P: Policy>(x: Vf<Self>, lnx: Vf<Self>) -> Vf<Self> {
        if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
            // determined empirically
            const X1: f32 = 9.1;
            const X2: f32 = 16.3;

            const B: f32 = 1.0 / (X2 - X1); // b
            const AB: f32 = X1 / (X2 - X1); // a*b where a=x1

            // combined into fma
            //let u1 = (x - Vf::splat(x1)) * Vf::splat(1.0 / (x2 - x1));
            let u1 = x.mul_sube(Vf::splat(B), Vf::splat(AB));

            // clamp
            let mut u1 = u1.min(Vf::ONE).max(Vf::ZERO);

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
            let mut res = u1.lerp_p::<P>(lnx - c, Vf::ZERO);

            if P::POLICY.check_overflow {
                res = res.cmp_lt(Vf::ZERO).select(Vf::NAN, res);
                res = res.cmp_eq(Vf::ZERO).select(Vf::NEG_INFINITY, res);
            }

            return res;
        }

        (Vf::ONE - (-x).exp_p::<P>()).ln_p::<P>()
    }

    #[inline(always)]
    fn erf<P: Policy>(x0: Vf<Self>) -> Vf<Self> {
        if const { P::POLICY.precision.eq(PrecisionPolicy::Reference) } {
            // Use erf(x) = 1 - erfc(x), since erfc has a good reference implementation
            return Vf::ONE - Self::erfc::<P>(x0);
        }

        let mut x = x0.abs();

        if P::POLICY.check_overflow {
            x = Vf::ONE - (Vf::ONE - x); // crush denormals
        }

        let y = match P::POLICY.precision {
            // 5 * 10^-4 accuracy
            PrecisionPolicy::Worst => {
                let t = x.poly_p::<P, _>(&[1.0, 0.278393, 0.230389, 0.000972, 0.078108]);
                let t2 = t * t;
                let t4 = t2 * t2;

                // 1 - 1/t4
                if R::HAS_APPROX_RCP {
                    // use raw approximate reciprocal when available
                    let y = t4.rcp();

                    // combine one 1/x newton iteration with 1-y for the final result
                    y.nmul_adde(t4.nmul_adde(y, Vf::TWO), Vf::ONE)
                } else {
                    // otherwise just do the reciprocal normally
                    Vf::ONE - t4.reciprocal_p::<P>()
                }
            }

            // 3 * 10^-7 accuracy
            PrecisionPolicy::Medium | PrecisionPolicy::Average => {
                let r = x.poly_p::<P, _>(&[
                    1.0,
                    0.0705230784,
                    0.0422820123,
                    0.0092705272,
                    0.0001520143,
                    0.0002765672,
                    0.0000430638,
                ]);

                let r2 = r * r;
                let r4 = r2 * r2;
                let r8 = r4 * r4;
                let r16 = r8 * r8;

                Vf::ONE - r16.reciprocal_p::<ExtraPrecision<P>>()
            }

            // this method is not used, but kept for reference since it's _supposedly_ more accurate,
            // but doesn't seem to be, maybe due to the reliance on the exponential function?
            // // 1.5 * 10^-7 accuracy
            // PrecisionPolicy::Average => {
            //     // 1 / (1 + p * x) where p = 0.3275911
            //     let t = x.mul_adde(Vf::splat(0.3275911), Vf::ONE).reciprocal_p::<P>();
            //
            //     let e = t * (-x * x).exp_p::<P>(); // e^(-x^2)
            //
            //     let y = t.poly_p::<P, _>(&[0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]);
            //
            //     y.nmul_adde(e, Vf::ONE)
            // }

            // 1.2 * 10^-7 accuracy
            PrecisionPolicy::Best => {
                // 1 / (1 + 1/2|x|)
                let t = x.mul_adde(Vf::HALF, Vf::ONE).reciprocal_p::<P>();

                let r0 = t.poly_p::<P, _>(&[
                    -1.26551223,
                    1.00002368,
                    0.37409196,
                    0.09678418,
                    -0.18628806,
                    0.27886807,
                    -1.13520398,
                    1.48851587,
                    -0.82215223,
                    0.17087277,
                ]);

                // 1 - r where r = e^(-x^2 + r0)
                t.nmul_adde(x.nmul_adde(x, r0).exp_p::<P>(), Vf::ONE)
            }
            PrecisionPolicy::Reference => unreachable!("Reference precision handled above"),
        };

        y.mul_sign(x0)
    }

    #[inline(always)]
    fn erfc<P: Policy>(x0: Vf<Self>) -> Vf<Self> {
        if const { P::POLICY.precision.lt(PrecisionPolicy::Reference) } {
            // Use erfc(x) = 1 - erf(x)
            return Vf::ONE - Self::erf::<P>(x0);
        }

        let x = x0.abs();
        let x2 = x0 * x0;

        let a0 = Vf::<R>::splat(0.56418958354775629);
        let a1 = x + Vf::splat(2.06955023132914151);

        let b0 = x2 + x.mul_adde(Vf::splat(2.06955023132914151), Vf::splat(5.80755613130301624));
        let b1 = x2 + x.mul_adde(Vf::splat(3.47954057099518960), Vf::splat(12.06166887286239555));

        let c0 = x2 + x.mul_adde(Vf::splat(3.47469513777439592), Vf::splat(12.07402036406381411));
        let c1 = x2 + x.mul_adde(Vf::splat(3.72068443960225092), Vf::splat(8.44319781003968454));

        let d0 = x2 + x.mul_adde(Vf::splat(4.00561509202259545), Vf::splat(9.30596659485887898));
        let d1 = x2 + x.mul_adde(Vf::splat(3.90225704029924078), Vf::splat(6.36161630953880464));

        let e0 = x2 + x.mul_adde(Vf::splat(5.16722705817812584), Vf::splat(9.12661617673673262));
        let e1 = x2 + x.mul_adde(Vf::splat(4.03296893109262491), Vf::splat(5.13578530585681539));

        let f0 = x2 + x.mul_adde(Vf::splat(5.95908795446633271), Vf::splat(9.19435612886969243));
        let f1 = x2 + x.mul_adde(Vf::splat(4.11240942957450885), Vf::splat(4.48640329523408675));

        let n = a0 * b0 * c0 * d0 * e0 * f0;
        let d = a1 * b1 * c1 * d1 * e1 * f1;

        let m = n / d;
        let e = (-x2).exp_p::<P>();

        // if x<0 then 2 - y, else y
        if R::HAS_TRUE_FMA {
            // exploit instruction-level parallelism if FMA is available
            x0.select_negative(m.nmul_add(e, Vf::TWO), m * e)
        } else {
            let y = m * e;

            x0.select_negative(Vf::TWO - y, y)
        }
    }

    #[inline(always)]
    fn erfinv<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        // (-1, 1) range
        let x = x.clamp(Vf::splat(-0.99999), Vf::splat(0.99999));

        let w = -x.nmul_adde(x, Vf::ONE).ln_p::<P>();

        let ge5 = w.cmp_ge(Vf::splat(5.0));

        let w0 = w - Vf::splat(2.5);
        let mut p0 = w0.poly_p::<P, _>(&[
            1.50140941,
            0.246640727,
            -0.00417768164,
            -0.00125372503,
            0.00021858087,
            -4.39150654e-06,
            -3.5233877e-06,
            3.43273939e-07,
            2.81022636e-08,
        ]);

        if P::POLICY.avoid_branching || crate::unlikely(ge5.any()) {
            let w1 = w.sqrt() - Vf::splat(3.0);
            let p1 = w1.poly_p::<P, _>(&[
                2.83297682,
                1.00167406,
                0.00943887047,
                -0.0076224613,
                0.00573950773,
                -0.00367342844,
                0.00134934322,
                0.000100950558,
                -0.000200214257,
            ]);

            p0 = ge5.select(p1, p0);
        }

        p0 * x
    }
}

#[inline(always)]
fn asin_f_internal<P: Policy, R: MathInternal<f32>, const ACOS: bool>(x: Vf<R>) -> Vf<R> {
    let xa = x.abs();

    if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
        /* Based on http://www.pouet.net/topic.php?which=9132&page=2
         * 85% accurate (ULP 0)
         * Examined 2130706434 values of acos:
         *   15.2000597 avg ULP diff, 4492 max ULP, 4.51803e-05 max error // without "denormal crush"
         * Examined 2130706434 values of acos:
         *   15.2007108 avg ULP diff, 4492 max ULP, 4.51803e-05 max error // with "denormal crush"
         */
        let mut m = xa.min(Vf::ONE); // clamp

        if P::POLICY.check_overflow {
            m = Vf::ONE - (Vf::ONE - m); // crush denormals
        }

        let a0 = (Vf::ONE - m).sqrt();
        let a1 = m.poly_p::<P, _>(&[FRAC_PI_2, -0.213300989, 0.077980478, -0.02164095]);

        if ACOS {
            if R::HAS_TRUE_FMA {
                // if FMA is available we can at least exploit instruction-level parallelism
                return x.select_negative(a0.nmul_add(a1, Vf::PI), a0 * a1);
            }

            let a = a0 * a1;
            return x.select_negative(Vf::PI - a, a);
        } else {
            // Max error is 4.51133e-05 (ULPS are higher because we are consistently off by a little amount).
            return a0.nmul_adde(a1, Vf::FRAC_PI_2).copysign(x);
        }
    }

    let is_big = xa.cmp_gt(Vf::<R>::HALF);

    // TODO: Branch to avoid sqrt?
    let x1 = Vf::<R>::HALF * (Vf::<R>::ONE - xa);
    let x3 = is_big.select(x1, xa * xa);
    let x4 = is_big.select(x1.sqrt(), xa);

    #[rustfmt::skip]
    let z = x3.poly_p::<P, _>(&[
        1.6666752422E-1,
        7.4953002686E-2,
        4.5470025998E-2,
        2.4181311049E-2,
        4.2163199048E-2,
    ])
    .mul_adde(x3 * x4, x4);

    let z1 = z + z;

    if ACOS {
        let z1 = x.select_negative(Vf::<R>::PI - z1, z1);
        let z2 = Vf::<R>::FRAC_PI_2 - z.mul_sign(x);

        is_big.select(z1, z2)
    } else {
        let z1 = Vf::<R>::FRAC_PI_2 - z1;

        is_big.select(z1, z).mul_sign(x)
    }
}

#[inline(always)]
fn pow2n_f<R: MathInternal<f32>>(n: Vf<R>) -> Vf<R> {
    let pow2_23 = Vf::<R>::splat(8388608.0);
    let bias = Vf::<R>::splat(127.0);

    (n + (bias + pow2_23)) << 23
}

#[inline(always)]
fn exp_f_internal<P: Policy, R: MathInternal<f32>, const MODE: u8>(x0: Vf<R>) -> Vf<R> {
    let mut x = x0;
    let mut r;

    let max_x = const {
        match MODE {
            EXP_MODE_EXP => 87.3,
            EXP_MODE_POW2 => 126.0,
            EXP_MODE_POW10 => 37.9,
            EXP_MODE_EXPH | EXP_MODE_EXPM1 => 89.0,
            _ => panic!("Invalid MODE for exp_f_internal"), // unreachable!() isn't const apparently
        }
    };

    let mut z = if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
        // https://stackoverflow.com/a/10792321 with a better 2^f fit
        // max. rel. error <= 1.73e-3 on [-87,88]

        // Compute t such that b^x = 2^t
        let t = match MODE {
            EXP_MODE_EXP | EXP_MODE_EXPH | EXP_MODE_EXPM1 => x * Vf::<R>::LOG2_E,
            EXP_MODE_POW10 => x * Vf::<R>::LOG10_2,
            EXP_MODE_POW2 => x,
            _ => unreachable!("Invalid MODE for exp_f_internal"),
        };

        let fi = t.floor();
        let f = t - fi;

        // if the exponent exceeds this method's limitations, then it's far outside of the valid range for exp
        let i: Vs<R> = fi.fast_cast();

        // polynomial approximation of 2^f
        let cf = f.poly_p::<P, _>(&[1.0, 0.695556856, 0.226173572, 0.0781455737]);

        // scale 2^f by 2^i
        let ci = Vs::<R>::from_bits(cf) + (i << 23);

        let z = Vf::<R>::from_bits(ci);

        match MODE {
            EXP_MODE_EXPH => z * Vf::<R>::HALF,
            EXP_MODE_EXPM1 => z - Vf::<R>::ONE,
            EXP_MODE_EXP | EXP_MODE_POW2 | EXP_MODE_POW10 => z,
            _ => unreachable!("Invalid MODE for exp_f_internal"),
        }
    } else {
        match MODE {
            EXP_MODE_POW2 => {
                r = x0.round();

                x -= r;
                x *= Vf::LN_2;
            }
            EXP_MODE_POW10 => {
                let log10_2_hi = Vf::<R>::splat(0.301025391); // log10(2) in two parts
                let log10_2_lo = Vf::<R>::splat(4.60503907E-6);

                r = (x0 * Vf::<R>::splat(LN_10 * LOG2_E)).round();

                x = r.nmul_adde(log10_2_hi, x); // x -= r * log10_2_hi;
                x = r.nmul_adde(log10_2_lo, x); // x -= r * log10_2_lo;
                x *= Vf::LN_10;
            }
            EXP_MODE_EXP | EXP_MODE_EXPM1 | EXP_MODE_EXPH => {
                let ln2f_hi = Vf::<R>::splat(0.693359375);
                let ln2f_lo = Vf::<R>::splat(-2.12194440e-4);

                r = (x0 * Vf::LOG2_E).round();

                x = r.nmul_adde(ln2f_hi, x); // x -= r * ln2f_hi;
                x = r.nmul_adde(ln2f_lo, x); // x -= r * ln2f_lo;

                if const { MODE == EXP_MODE_EXPH } {
                    r -= Vf::ONE;
                }
            }
            _ => unreachable!("Invalid MODE for exp_f_internal"),
        }

        let mut z = x
            .poly_p::<P, _>(&[1.0 / 2.0, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0, 1.0 / 720.0, 1.0 / 5040.0])
            .mul_adde(x * x, x);

        let n2 = pow2n_f::<R>(r);

        match MODE {
            EXP_MODE_EXPM1 => z.mul_adde(n2, n2 - Vf::ONE),
            _ => z.mul_adde(n2, n2), // (z + 1.0f) * n2
        }
    };

    if const { P::POLICY.check_overflow } {
        let in_range = x0.abs().cmp_lt(Vf::<R>::splat(max_x)) & x0.is_finite();

        if crate::likely(in_range.all()) {
            return z;
        }

        let underflow_value = match MODE {
            EXP_MODE_EXPM1 => Vf::<R>::NEG_ONE,
            _ => Vf::<R>::ZERO,
        };

        r = x0.select_negative(underflow_value, Vf::INFINITY);
        z = in_range.select(z, r);
        z = x0.is_nan().select(x0, z);
    }

    z
}

#[inline(always)]
fn fraction2<R: MathInternal<f32>>(x: Vf<R>) -> Vf<R> {
    // set exponent to 0 + bias
    (x & Vf::<R>::splat(f32::from_bits(0x007FFFFF))) | Vf::<R>::splat(f32::from_bits(0x3F000000))
}

#[inline(always)]
fn exponent<R: MathInternal<f32>>(x: Vf<R>) -> Vs<R> {
    // shift out sign, extract exp, subtract bias
    Vs::<R>::from_bits((Vu::<R>::from_bits(x) << 1) >> 24) - Vs::<R>::splat(0x7F)
}

#[inline(always)]
fn ln_2_internal<P: Policy, R: MathInternal<f32>>(x: Vf<R>) -> Vf<R> {
    if const { P::POLICY.precision.eq(PrecisionPolicy::Worst) } {
        // // https://github.com/nadavrot/fast_log/blob/83bd112c330976c291300eaa214e668f809367ab/src/log_approx.cc#L47
        // return fraction2::<R>(x).poly_p::<P, _>(&[-3.21430967, 6.30371424, -4.42852392, 1.33755322])
        //     + (exponent::<R>(x) + Vs::<R>::ONE).cast();

        // https://github.com/romeric/fastapprox/blob/ccc534400ec3e0f67de4eafb53377334962d9db6/fastapprox/src/fastonebigheader.h#L384
        // between 1e-4 and 1000, avg error: 0.00536, max error 0.0573 at 31.999878
        return Vf::from(Vs::<R>::from_bits(x)).mul_sube(Vf::splat(1.1920928955078125e-7), Vf::splat(126.94269504));
    }

    ln_f_internal::<P, R, false>(x) * Vf::LOG2_E
}

#[inline(always)]
fn ln_10_internal<P: Policy, R: MathInternal<f32>>(x: Vf<R>) -> Vf<R> {
    if const { P::POLICY.precision.eq(PrecisionPolicy::Worst) } {
        // ln(x) * LOG10_E
        // between 1e-4 and 1000, avg error: 0.00212, max error 0.0173 at 31.999878
        return Vf::from(Vs::<R>::from_bits(x)).mul_sube(Vf::splat(3.5885571887588505e-8), Vf::splat(38.213558906));
    }

    ln_f_internal::<P, R, false>(x) * Vf::LOG10_E
}

#[inline(always)]
fn ln_f_internal<P: Policy, R: MathInternal<f32>, const P1: bool>(x0: Vf<R>) -> Vf<R> {
    if const { P::POLICY.precision.eq(PrecisionPolicy::Worst) } {
        let x1 = if P1 { x0 + Vf::ONE } else { x0 };

        // https://github.com/romeric/fastapprox/blob/ccc534400ec3e0f67de4eafb53377334962d9db6/fastapprox/src/fastonebigheader.h#L393
        // between 1e-4 and 1000, avg error: 0.00536, max error 0.0397 at 3.9999847
        return Vf::from(Vs::<R>::from_bits(x1)).mul_sube(Vf::splat(8.2629582881927490e-8), Vf::splat(87.989971088));
    }

    if const { P::POLICY.precision.eq(PrecisionPolicy::Medium) } {
        // https://stackoverflow.com/a/39822314/2083075
        // natural log on [0x1.f7a5ecp-127, 0x1.fffffep127]. Maximum relative error 9.4529e-5

        let a = Vs::<R>::from_bits(x0);
        let e = (a - Vs::<R>::splat(0x3f2aaaab)) & Vs::<R>::splat(0xff800000u32 as i32);
        let i = Vf::from(e) * Vf::splat(1.19209290e-7);
        let mut f = Vf::from_bits(a - e);

        if !P1 {
            f -= Vf::ONE;
        }

        let s = f * f;

        /* Compute log1p(f) for f in [-1/3, 1/3] */
        let r = f.mul_adde(Vf::splat(0.230836749), Vf::splat(-0.279208571)); // 0x1.d8c0f0p-3, -0x1.1de8dap-2
        let t = f.mul_adde(Vf::splat(0.331826031), Vf::splat(-0.498910338)); // 0x1.53ca34p-2, -0x1.fee25ap-2
        let r = r.mul_adde(s, t);
        let r = r.mul_adde(s, f);
        let r = i.mul_adde(Vf::splat(0.693147182), r); // 0x1.62e430p-1 // log(2)

        return r;
    }

    let ln2f_hi = Vf::splat(0.693359375);
    let ln2f_lo = Vf::splat(-2.12194440E-4);

    let x1 = if P1 { x0 + Vf::ONE } else { x0 };

    let mut x = fraction2::<R>(x1);
    let mut e = exponent::<R>(x1);

    let blend = x.cmp_gt(Vf::splat(SQRT_2 * 0.5));

    x = blend.select(x, x + x); // x.conditional_add(x, !blend)
    e = blend.select(e + Vs::<R>::ONE, e); // e.conditional_add(Vs::<R>::ONE, blend)

    let fe: Vf<R> = e.cast();

    let xp1 = x - Vf::ONE;

    x = if P1 {
        // log(x+1). Avoid loss of precision when adding 1 and later subtracting 1 if exponent = 0
        e.cmp_eq(Vs::<R>::ZERO).select(x0, xp1)
    } else {
        xp1 // log(x). Expand around 1.0
    };

    let x2 = x * x;
    let mut res = x.poly_p::<P, _>(&[
        0.0, // multiply all by x
        3.3333331174E-1,
        -2.4999993993E-1,
        2.0000714765E-1,
        -1.6668057665E-1,
        1.4249322787E-1,
        -1.2420140846E-1,
        1.1676998740E-1,
        -1.1514610310E-1,
        7.0376836292E-2,
    ]);

    res = fe.mul_adde(ln2f_lo, res.mul_adde(x2, x2.nmul_adde(Vf::HALF, x)));
    res = fe.mul_adde(ln2f_hi, res);

    if const { !P::POLICY.check_overflow } {
        return res;
    }

    let overflow = !x1.is_finite();
    let underflow = x1.cmp_lt(Vf::<R>::splat(1.17549435e-38));

    if !P::POLICY.avoid_branching && crate::likely((overflow | underflow).none()) {
        return res;
    }

    res = underflow.select(Vf::NAN, res); // x1 < 0 gives NAN
    res = x1.is_zero_or_subnormal().select(Vf::NEG_INFINITY, res); // x1 == 0 gives -INF
    res = overflow.select(x1, res); // INF or NAN goes through
    res = (x1.is_infinite() & x1.is_negative()).select(Vf::NAN, res); // -INF gives NAN

    res
}
