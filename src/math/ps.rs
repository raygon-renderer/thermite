use super::{poly::*, *};

use core::f32::consts::{
    FRAC_1_PI, FRAC_2_PI, FRAC_PI_2, FRAC_PI_4, LN_10, LN_2, LOG10_2, LOG10_E, LOG2_E, PI, SQRT_2,
};

const EULERS_CONSTANT: f32 = 5.772156649015328606065120900824024310e-01;
const LN_PI: f32 = 1.1447298858494001741434273513530587116472948129153115715136230714;
const SQRT_E: f32 = 1.6487212707001281468486507878141635716537761007101480115750793116;

impl<S: Simd> SimdVectorizedMathInternal<S> for f32
where
    <S as Simd>::Vf32: SimdFloatVector<S, Element = f32>,
{
    type Vf = <S as Simd>::Vf32;

    const __EPSILON: Self = f32::EPSILON;
    const __SQRT_EPSILON: Self = 0.0003452669836517821464776144458809047877858776827733458406716232;
    const __PI: Self = core::f32::consts::PI;
    const __DIGITS: u32 = f32::MANTISSA_DIGITS;

    #[inline(always)]
    fn sin_cos<P: Policy>(xx: Self::Vf) -> (Self::Vf, Self::Vf) {
        if P::POLICY.precision == PrecisionPolicy::Worst {
            // https://stackoverflow.com/a/28050328/2083075 with modifications
            #[inline(always)]
            fn fast_cosine<S: Simd>(mut x: Vf32<S>, sine: bool) -> Vf32<S> {
                let quarter = Vf32::<S>::splat(0.25);
                let half = Vf32::<S>::splat(0.5);

                // encourage instruction-level parallelism
                if sine {
                    x = (x - half) - x.floor();
                } else {
                    x = (x - quarter) - (x + quarter).floor();
                }

                // rearrange for FMA, no chance of overflow since x is (-0.5, 0.5) here
                //x *= Vf32::<S>::splat(16.0) * (x.abs() - Vf32::<S>::splat(0.5));
                x *= x.abs().mul_sube(Vf32::<S>::splat(16.0), Vf32::<S>::splat(8.0));

                let c = Vf32::<S>::splat(0.225);
                x.mul_adde(x.abs().mul_sube(c, c), x)
            }

            let x = xx * Vf32::<S>::splat(FRAC_1_PI * 0.5);

            let sine = fast_cosine::<S>(x, true);
            let cosine = fast_cosine::<S>(x, false);

            return (sine, cosine);
        }

        let dp1f = Vf32::<S>::splat(0.78515625 * 2.0);
        let dp2f = Vf32::<S>::splat(2.4187564849853515625E-4 * 2.0);
        let dp3f = Vf32::<S>::splat(3.77489497744594108E-8 * 2.0);
        let xa: Vf32<S> = xx.abs();

        let y: Vf32<S> = (xa * Vf32::<S>::splat(FRAC_2_PI)).round();
        let q: Vu32<S> = y.cast_to::<Vi32<S>>().into_bits(); // cast to signed (faster), then transmute to unsigned

        // Reduce by extended precision modular arithmetic
        // x = ((xa - y * DP1F) - y * DP2F) - y * DP3F;
        let x = y.nmul_add(dp3f, y.nmul_add(dp2f, y.nmul_add(dp1f, xa)));

        // Taylor expansion of sin and cos, valid for -pi/4 <= x <= pi/4
        let x2 = x * x;
        let mut s = x2
            .poly_p::<P>(&[-1.6666654611E-1, 8.3321608736E-3, -1.9515295891E-4])
            .mul_adde(x2 * x, x);

        let mut c = x2
            .poly_p::<P>(&[4.166664568298827E-2, -1.388731625493765E-3, 2.443315711809948E-5])
            .mul_adde(x2 * x2, Vf32::<S>::splat(0.5).nmul_adde(x2, Vf32::<S>::one()));

        // swap sin and cos if odd quadrant
        let swap = (q & Vu32::<S>::one()).ne(Vu32::<S>::zero());

        if P::POLICY.check_overflow {
            let overflow = q.gt(Vu32::<S>::splat(0x2000000)) & xa.is_finite().cast_to(); // q big if overflow

            s = overflow.select(Vf32::<S>::zero(), s);
            c = overflow.select(Vf32::<S>::one(), c);
        }

        let sin1 = swap.select(c, s);
        let cos1 = swap.select(s, c);

        let signsin = Vf32::<S>::from_bits(q << 30) ^ xx;
        let signcos = Vf32::<S>::from_bits(((q + Vu32::<S>::one()) & Vu32::<S>::splat(2)) << 30);

        // combine signs
        (sin1.combine_sign(signsin), cos1 ^ signcos)
    }

    #[inline(always)]
    fn sinh<P: Policy>(x0: Self::Vf) -> Self::Vf {
        let x = x0.abs();

        let x_small = x.le(Vf32::<S>::one());

        let mut y1 = unsafe { Vf32::<S>::undefined() };
        let mut y2 = unsafe { Vf32::<S>::undefined() };

        // use bitmask directly to avoid two calls
        let bitmask = x_small.bitmask();

        // if not all are small
        if P::POLICY.avoid_branching || !bitmask.all() {
            y2 = x.exph_p::<P>();
            y2 -= Vf32::<S>::splat(0.25) / y2;

            if P::POLICY.avoid_precision_branches() {
                return y2.combine_sign(x0);
            }
        }

        // if any are small
        if P::POLICY.avoid_branching || bitmask.any() {
            let x2 = x * x;
            y1 = x2
                .poly_p::<P>(&[1.66667160211E-1, 8.33028376239E-3, 2.03721912945E-4])
                .mul_adde(x2 * x, x);
        }

        x_small.select(y1, y2).combine_sign(x0)
    }

    #[inline(always)]
    fn tanh<P: Policy>(x0: Self::Vf) -> Self::Vf {
        let one = Vf32::<S>::one();

        let x = x0.abs();
        let x_small = x.le(Vf32::<S>::splat(0.625));

        let mut y1 = unsafe { Vf32::<S>::undefined() };
        let mut y2 = unsafe { Vf32::<S>::undefined() };

        // use bitmask directly to avoid two calls
        let bitmask = x_small.bitmask();

        // if not all are small
        if P::POLICY.avoid_branching || !bitmask.all() {
            y2 = (x + x).exp_p::<P>();
            // originally (1 - 2/(y2 + 1)), but doing it this way avoids
            // loading 2.0 and encourages slight instruction-level parallelism
            y2 = (y2 - one) / (y2 + one);

            if P::POLICY.check_overflow {
                y2 = x.gt(Vf32::<S>::splat(44.4)).select(one, y2);
            }

            if P::POLICY.avoid_precision_branches() {
                return y2.combine_sign(x0);
            }
        }

        // if any are small
        if P::POLICY.avoid_branching || bitmask.any() {
            let x2 = x * x;

            y1 = x2
                .poly_p::<P>(&[
                    -3.33332819422E-1,
                    1.33314422036E-1,
                    -5.37397155531E-2,
                    2.06390887954E-2,
                    -5.70498872745E-3,
                ])
                .mul_adde(x2 * x, x);
        }

        x_small.select(y1, y2).combine_sign(x0)
    }

    #[inline(always)]
    fn asin<P: Policy>(x: Self::Vf) -> Self::Vf {
        asin_f_internal::<S, P>(x, false)
    }

    #[inline(always)]
    fn acos<P: Policy>(x: Self::Vf) -> Self::Vf {
        asin_f_internal::<S, P>(x, true)
    }

    #[inline(always)]
    fn atan<P: Policy>(y: Self::Vf) -> Self::Vf {
        let t = y.abs();

        let not_small = t.ge(Vf32::<S>::splat(SQRT_2 - 1.0)); // t >= tan  pi/8
        let not_big = t.le(Vf32::<S>::splat(SQRT_2 + 1.0)); // t <= tan 3pi/8

        let s = not_big.select(Vf32::<S>::splat(FRAC_PI_4), Vf32::<S>::splat(FRAC_PI_2)) & not_small.value(); // select(not_small, s, 0.0);

        // small:  z = t / 1.0;
        // medium: z = (t-1.0) / (t+1.0);
        // big:    z = -1.0 / t;

        // this trick avoids having to place a zero in any register
        let a = (not_big.value() & t) + (not_small.value() & Vf32::<S>::neg_one());
        let b = (not_big.value() & Vf32::<S>::one()) + (not_small.value() & t);

        let z = a / b;
        let z2 = z * z;

        z2.poly_p::<P>(&[-3.33329491539E-1, 1.99777106478E-1, -1.38776856032E-1, 8.05374449538E-2])
            .mul_adde(z2 * z, z + s)
            .combine_sign(y)
    }

    #[inline(always)]
    fn atan2<P: Policy>(y: Self::Vf, x: Self::Vf) -> Self::Vf {
        let neg_one = Vf32::<S>::neg_one();
        let zero = Vf32::<S>::zero();

        let x1 = x.abs();
        let y1 = y.abs();

        let swap_xy = y1.gt(x1);

        let mut x2 = swap_xy.select(y1, x1);
        let mut y2 = swap_xy.select(x1, y1);

        if P::POLICY.check_overflow {
            let both_infinite = (x.is_infinite() & y.is_infinite());

            if unlikely!(both_infinite.any()) {
                x2 = both_infinite.select(x2 & neg_one, x2); // get 1.0 with the sign of x
                y2 = both_infinite.select(y2 & neg_one, y2); // get 1.0 with the sign of y
            }
        }

        // x = y = 0 will produce NAN. No problem, fixed below
        let t = y2 / x2;

        // small:  z = t / 1.0;
        // medium: z = (t-1.0) / (t+1.0);
        let not_small = t.ge(Vf32::<S>::splat(SQRT_2 - 1.0));

        let a = t + (not_small.value() & neg_one);
        let b = Vf32::<S>::one() + (not_small.value() & t);

        let s = not_small.value() & Vf32::<S>::splat(FRAC_PI_4);

        let z = a / b;
        let z2 = z * z;

        let mut re = z2
            .poly_p::<P>(&[-3.33329491539E-1, 1.99777106478E-1, -1.38776856032E-1, 8.05374449538E-2])
            .mul_adde(z2 * z, z + s);

        re = swap_xy.select(Vf32::<S>::splat(FRAC_PI_2) - re, re);
        re = (x | y).eq(zero).select(zero, re); // atan2(0,+0) = 0 by convention
        re = x.select_negative(Vf32::<S>::splat(PI) - re, re); // also for x = -0.

        re
    }

    #[inline(always)]
    fn asinh<P: Policy>(x0: Self::Vf) -> Self::Vf {
        let x = x0.abs();
        let x2 = x0 * x0;

        let x_small = x.le(Vf32::<S>::splat(0.51));

        let mut y1 = unsafe { Vf32::<S>::undefined() };
        let mut y2 = unsafe { Vf32::<S>::undefined() };

        let bitmask = x_small.bitmask();

        if P::POLICY.avoid_branching || !bitmask.all() {
            y2 = ((x2 + Vf32::<S>::one()).sqrt() + x).ln_p::<P>();

            if P::POLICY.check_overflow {
                let x_huge = x.gt(Vf32::<S>::splat(1e10));

                if P::POLICY.avoid_precision_branches() || unlikely!(x_huge.any()) {
                    y2 = x_huge.select(x.ln_p::<P>() + Vf32::<S>::splat(LN_2), y2);
                }
            }

            if P::POLICY.avoid_precision_branches() {
                return y2.combine_sign(x0);
            }
        }

        if P::POLICY.avoid_branching || bitmask.any() {
            y1 = x2
                .poly_p::<P>(&[-1.6666288134E-1, 7.4847586088E-2, -4.2699340972E-2, 2.0122003309E-2])
                .mul_adde(x2 * x, x);
        }

        x_small.select(y1, y2).combine_sign(x0)
    }

    #[inline(always)]
    fn acosh<P: Policy>(x0: Self::Vf) -> Self::Vf {
        let one = Vf32::<S>::one();

        let x1 = x0 - one;

        let x_small = x1.lt(Vf32::<S>::splat(0.49)); // use Pade approximation if abs(x-1) < 0.5

        let mut y1 = unsafe { Vf32::<S>::undefined() };
        let mut y2 = unsafe { Vf32::<S>::undefined() };

        let bitmask = x_small.bitmask();

        // if not all are small
        if P::POLICY.avoid_branching || !bitmask.all() {
            y2 = (x0.mul_sube(x0, one).sqrt() + x0).ln_p::<P>();

            if P::POLICY.check_overflow {
                let x_huge = x1.gt(Vf32::<S>::splat(1e10));

                if P::POLICY.avoid_precision_branches() || unlikely!(x_huge.any()) {
                    y2 = x_huge.select(x0.ln_p::<P>() + Vf32::<S>::splat(LN_2), y2);
                }
            }

            if P::POLICY.avoid_precision_branches() {
                return y2;
            }
        }

        // if any are small
        if P::POLICY.avoid_branching || bitmask.any() {
            y1 = x1.sqrt()
                * x1.poly_p::<P>(&[
                    1.4142135263E0,
                    -1.1784741703E-1,
                    2.6454905019E-2,
                    -7.5272886713E-3,
                    1.7596881071E-3,
                ]);

            if P::POLICY.check_overflow {
                // result is NaN if less-than 1
                y1 = x0.lt(one).select(Vf32::<S>::nan(), y1);
            }
        }

        x_small.select(y1, y2)
    }

    #[inline(always)]
    fn atanh<P: Policy>(x0: Self::Vf) -> Self::Vf {
        let x = x0.abs();

        let x_small = x.lt(Vf32::<S>::splat(0.5));

        let mut y1 = unsafe { Vf32::<S>::undefined() };
        let mut y2 = unsafe { Vf32::<S>::undefined() };

        let bitmask = x_small.bitmask();

        if P::POLICY.avoid_branching || !bitmask.all() {
            let one = Vf32::<S>::one();

            y2 = ((one + x) / (one - x)).ln_p::<P>() * Vf32::<S>::splat(0.5);

            if P::POLICY.check_overflow {
                let y3 = x.eq(one).select(Vf32::<S>::infinity(), Vf32::<S>::nan());
                y2 = x.ge(one).select(y3, y2);
            }

            if P::POLICY.avoid_precision_branches() {
                return y2.combine_sign(x0);
            }
        }

        if P::POLICY.avoid_branching || bitmask.any() {
            let x2 = x * x;

            y1 = x2
                .poly_p::<P>(&[
                    3.33337300303E-1,
                    1.99782164500E-1,
                    1.46691431730E-1,
                    8.24370301058E-2,
                    1.81740078349E-1,
                ])
                .mul_adde(x2 * x, x);
        }

        x_small.select(y1, y2).combine_sign(x0)
    }

    #[inline(always)]
    fn exp<P: Policy>(x: Self::Vf) -> Self::Vf {
        exp_f_internal::<S, P>(x, ExpMode::Exp)
    }

    #[inline(always)]
    fn exph<P: Policy>(x: Self::Vf) -> Self::Vf {
        exp_f_internal::<S, P>(x, ExpMode::Exph)
    }

    #[inline(always)]
    fn exp2<P: Policy>(x: Self::Vf) -> Self::Vf {
        exp_f_internal::<S, P>(x, ExpMode::Pow2)
    }

    #[inline(always)]
    fn exp10<P: Policy>(x: Self::Vf) -> Self::Vf {
        exp_f_internal::<S, P>(x, ExpMode::Pow10)
    }

    #[inline(always)]
    fn exp_m1<P: Policy>(x: Self::Vf) -> Self::Vf {
        exp_f_internal::<S, P>(x, ExpMode::Expm1)
    }

    #[inline(always)]
    fn cbrt<P: Policy>(x: Self::Vf) -> Self::Vf {
        let b1 = Vu32::<S>::splat(709958130); // B1 = (127-127.0/3-0.03306235651)*2**23
        let b2 = Vu32::<S>::splat(642849266); // B2 = (127-127.0/3-24/3-0.03306235651)*2**23
        let m = Vu32::<S>::splat(0x7fffffff); // u32::MAX >> 1

        let x1p24 = x * Vf32::<S>::splat(f32::from_bits(0x4b800000)); // 0x1p24f === 2 ^ 24

        let hx0 = x.into_bits() & m;

        let x_small = hx0.lt(Vu32::<S>::splat(0x00800000));

        let xs = x_small.select(x1p24, x);
        let b = x_small.select(b2, b1);

        let mut ui = xs.into_bits();
        let mut hx = ui & m;

        // TODO: Fix this when stable isn't broken
        hx = <Vu32<S> as Div<Divider<u32>>>::div(hx, Divider::u32(3)) + b;

        ui &= Vu32::<S>::splat(0x80000000);
        ui |= hx;

        let mut t = Vf32::<S>::from_bits(ui);

        if P::POLICY.precision >= PrecisionPolicy::Best || !S::INSTRSET.has_true_fma() {
            let mut td = t.cast_to::<Vf64<S>>();
            let xd = x.cast_to::<Vf64<S>>();

            // First iteration accurate to 16 bits, second iteration to 47 bits.
            for _ in 0..2 {
                let r = td * td * td;
                let rxd = xd + r;
                td *= (xd + rxd) / (r + rxd);
            }

            t = <Vf32<S> as SimdFromCast<S, Vf64<S>>>::from_cast(td);
        } else {
            let two = Vf32::<S>::splat(2.0);

            // couple iterations of Newton's method
            // This isn't perfect, as it's only limited to single-precision,
            // but the fused multiply-adds helps
            for _ in 0..2 {
                let t3 = t * t * t;
                t *= two.mul_add(x, t3) / two.mul_add(t3, x); // try to use extended precision where possible
            }
        }

        if !P::POLICY.check_overflow {
            // use float cmp and blend here to avoid domain change
            return x.eq(Vf32::<S>::zero()).select(x, t);
        }

        // cbrt(NaN,INF,+-0) is itself
        (hx0.gt(Vu32::<S>::splat(0x7f800000)) | hx0.eq(Vu32::<S>::zero())).select(x, t)
    }

    #[inline(always)]
    fn powf<P: Policy>(x0: Self::Vf, y: Self::Vf) -> Self::Vf {
        if P::POLICY.precision == PrecisionPolicy::Worst {
            return (x0.log2_p::<P>() * y).exp2_p::<P>();
        }

        // define constants
        let ln2f_hi = Vf32::<S>::splat(0.693359375); // log(2), split in two for extended precision
        let ln2f_lo = Vf32::<S>::splat(-2.12194440e-4);
        let log2e = Vf32::<S>::splat(LOG2_E); // 1/ln(2)
        let ln2 = Vf32::<S>::splat(LN_2);

        let zero = Vf32::<S>::zero();
        let one = Vf32::<S>::one();
        let half = Vf32::<S>::splat(0.5);

        let x1 = x0.abs();

        let mut x = fraction2::<S>(x1);

        let blend = x.gt(Vf32::<S>::splat(SQRT_2 * 0.5));

        // reduce range of x = +/- sqrt(2)/2
        x += !blend.value() & x;
        x -= one;

        // Taylor expansion, high precision
        let x2 = x * x;

        // logarithm expansion
        let mut lg1 = x.poly_p::<P>(&[
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

        let ef = <Vf32<S> as SimdFromCast<S, Vi32<S>>>::from_cast(exponent::<S>(x1)) + (blend.value() & one);

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
            .poly_p::<P>(&[1.0 / 2.0, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0, 1.0 / 720.0, 1.0 / 5040.0])
            .mul_adde(x * x, x + one);

        // contributions to exponent
        let ee = e1 + e2 + e3;
        let ei = ee.cast_to::<Vi32<S>>();

        // biased exponent of result:
        let ej = ei + Vi32::<S>::from_bits(z.into_bits()) >> 23;

        // add exponent by signed integer addition
        let mut z = Vf32::<S>::from_bits((Vi32::<S>::from_bits(z.into_bits()) + (ei << 23)).into_bits());

        if !P::POLICY.check_overflow {
            return z;
        }

        // check exponent for overflow and underflow
        let overflow = Vf32::<S>::from_cast_mask(ej.ge(Vi32::<S>::splat(0x0FF))) | ee.gt(Vf32::<S>::splat(300.0));
        let underflow = Vf32::<S>::from_cast_mask(ej.le(Vi32::<S>::splat(0x000))) | ee.lt(Vf32::<S>::splat(-300.0));

        // check for special cases
        let xfinite = x0.is_finite();
        let yfinite = y.is_finite();
        let efinite = ee.is_finite();

        let xzero = x0.is_zero_or_subnormal();
        let xsign = x0.is_negative();

        if unlikely!((overflow | underflow).any()) {
            z = underflow.select(zero, z);
            z = overflow.select(Vf32::<S>::infinity(), z);
        }

        let yzero = y.eq(zero);
        let yneg = y.lt(zero);

        // pow_case_x0
        z = xzero.select(yneg.select(Vf32::<S>::infinity(), yzero.select(one, zero)), z);

        let mut yodd = zero;

        if xsign.any() {
            let yint = y.eq(y.round());
            yodd = y << 31;

            let z1 = yint.select(z | yodd, x0.eq(zero).select(z, Vf32::<S>::nan()));

            yodd = yint.select(yodd, zero);

            z = xsign.select(z1, z);
        }

        let not_special = (xfinite & yfinite & (efinite | xzero));

        if likely!(not_special.all()) {
            return z; // fast return
        }

        // handle special error cases: y infinite
        let z1 = (yfinite & efinite).select(
            z,
            x1.eq(one)
                .select(one, (x1.gt(one) ^ y.is_negative()).select(Vf32::<S>::infinity(), zero)),
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
    fn ln<P: Policy>(x: Self::Vf) -> Self::Vf {
        ln_f_internal::<S, P>(x, false)
    }

    #[inline(always)]
    fn ln_1p<P: Policy>(x: Self::Vf) -> Self::Vf {
        ln_f_internal::<S, P>(x, true)
    }

    #[inline(always)]
    fn log2<P: Policy>(x: Self::Vf) -> Self::Vf {
        ln_f_internal::<S, P>(x, false) * Vf32::<S>::splat(LOG2_E)
    }

    #[inline(always)]
    fn log10<P: Policy>(x: Self::Vf) -> Self::Vf {
        ln_f_internal::<S, P>(x, false) * Vf32::<S>::splat(LOG10_E)
    }

    #[inline(always)]
    fn erf<P: Policy>(x: Self::Vf) -> Self::Vf {
        x * (x * x).poly_p::<P>(&[
            1.128379165726710e+0,
            -3.761262582423300e-1,
            1.128358514861418e-1,
            -2.685381193529856e-2,
            5.188327685732524e-3,
            -8.010193625184903e-4,
            7.853861353153693e-5,
        ])
    }

    #[inline(always)]
    fn erfinv<P: Policy>(y: Self::Vf) -> Self::Vf {
        /*
            Approximating the erfinv function, Mike Giles
            https://people.maths.ox.ac.uk/gilesm/files/gems_erfinv.pdf
        */
        let one = Vf32::<S>::one();

        let a = y.abs();

        let w = -a.nmul_adde(a, one).ln_p::<P>();

        let mut p0 = (w - Vf32::<S>::splat(2.5)).poly_p::<P>(&[
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

        let w_big = w.ge(Vf32::<S>::splat(5.0)); // at around |x| > 0.99662533231, so unlikely

        // avoids a costly sqrt and polynomial if false
        if unlikely!(w_big.any()) {
            let mut p1 = (w.sqrt() - Vf32::<S>::splat(3.0)).poly_p::<P>(&[
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

            if P::POLICY.check_overflow {
                p1 = a.eq(one).select(Vf32::<S>::infinity(), p1); // erfinv(x == 1) = inf
                p1 = a.gt(one).select(Vf32::<S>::nan(), p1); // erfinv(x > 1) = NaN
            }

            p0 = w_big.select(p1, p0);
        }

        p0 * y
    }

    #[inline(always)]
    fn next_float<P: Policy>(x: Self::Vf) -> Self::Vf {
        let i1 = Vu32::<S>::one();

        let v = x.eq(Vf32::<S>::neg_zero()).select(Vf32::<S>::zero(), x);

        let bits = v.into_bits();
        let finite = Vf32::<S>::from_bits(v.ge(Vf32::<S>::zero()).select(bits + i1, bits - i1));

        if !P::POLICY.check_overflow {
            return finite;
        }

        x.eq(Vf32::<S>::infinity()).select(x, finite)
    }

    #[inline(always)]
    fn prev_float<P: Policy>(x: Self::Vf) -> Self::Vf {
        let i1 = Vu32::<S>::one();

        let v = x.eq(Vf32::<S>::zero()).select(Vf32::<S>::neg_zero(), x);

        let bits = v.into_bits();
        let finite = Vf32::<S>::from_bits(v.gt(Vf32::<S>::zero()).select(bits - i1, bits + i1));

        if !P::POLICY.check_overflow {
            return finite;
        }

        x.eq(Vf32::<S>::neg_infinity()).select(x, finite)
    }

    #[inline(always)]
    fn tgamma<P: Policy>(mut z: Self::Vf) -> Self::Vf {
        let zero = Vf32::<S>::zero();
        let one = Vf32::<S>::one();
        let half = Vf32::<S>::splat(0.5);
        let quarter = Vf32::<S>::splat(0.25);
        let pi = Vf32::<S>::splat(PI);

        let orig_z = z;

        let is_neg = z.is_negative();
        let mut reflected = Mask::falsey();

        let mut res = one;

        'goto_positive: while is_neg.any() {
            reflected = z.le(Vf32::<S>::splat(-20.0));

            let mut refl_res = unsafe { Vf32::<S>::undefined() };

            // sine is expensive, so branch for it.
            if P::POLICY.avoid_precision_branches() || unlikely!(reflected.any()) {
                refl_res = <Self as SimdVectorizedMathInternal<S>>::sin_pix::<P>(z);

                // If not branching, all negative values are reflected
                if P::POLICY.avoid_precision_branches() {
                    reflected = is_neg;

                    res = reflected.select(refl_res, res);
                    z = z.conditional_neg(reflected);

                    break 'goto_positive;
                }

                // NOTE: I chose not to use a bitmask here, because some bitmasks can be
                // one extra instruction than the raw call to `all` again, and since z <= -20 is so rare,
                // that extra instruction is not worth it.
                if reflected.all() {
                    res = refl_res;
                    z = -z;

                    break 'goto_positive;
                }
            }

            let mut mod_z = z;
            let mut is_neg = is_neg;

            // recursively apply Γ(z+1)/z
            while is_neg.any() {
                res = is_neg.select(res / mod_z, res);
                mod_z = mod_z.conditional_add(one, is_neg);
                is_neg = mod_z.is_negative();
            }

            z = reflected.select(-z, mod_z);
            res = reflected.select(refl_res, res);

            break 'goto_positive;
        }

        // label
        //positive:

        // Integers

        let mut z_int = Mask::falsey();
        let mut fact_res = one;

        if P::POLICY.precision > PrecisionPolicy::Worst {
            let zf = z.floor();
            z_int = zf.eq(z);

            let bitmask = z_int.bitmask();

            if unlikely!(bitmask.any()) {
                let mut j = one;
                let mut k = j.lt(zf);

                while k.any() {
                    fact_res = k.select(fact_res * j, fact_res);
                    j += one;
                    k = j.lt(zf);
                }

                // Γ(-int) = NaN for poles
                fact_res = is_neg.select(Vf32::<S>::nan(), fact_res);
                // approaching zero from either side results in +/- infinity
                fact_res = orig_z.eq(zero).select(Vf32::<S>::infinity().copysign(orig_z), fact_res);

                if bitmask.all() {
                    return fact_res;
                }
            }
        }

        // Full

        let gh = Vf32::<S>::splat(LANCZOS_G - 0.5);

        let lanczos_sum = z.poly_rational_p::<P>(LANCZOS_P, LANCZOS_Q);

        let zgh = z + gh;
        let lzgh = zgh.ln_p::<P>();

        // (z * lzfg) > ln(f32::MAX)
        let very_large = (z * lzgh).gt(Vf32::<S>::splat(
            88.722839053130621324601674778549183073943430402325230485234240247,
        ));

        // only compute powf once
        let h = zgh.powf_p::<P>(very_large.select(z.mul_sube(half, quarter), z - half));

        // save a couple cycles by avoiding this division, but worst-case precision is slightly worse
        let denom = if P::POLICY.precision >= PrecisionPolicy::Best {
            lanczos_sum / zgh.exp_p::<P>()
        } else {
            lanczos_sum * (-zgh).exp_p::<P>()
        };

        let normal_res = very_large.select(h * h, h) * denom;

        // Tiny
        if P::POLICY.precision >= PrecisionPolicy::Best {
            let is_tiny = z.lt(Vf32::<S>::splat(
                <Self as SimdVectorizedMathInternal<S>>::__SQRT_EPSILON,
            ));
            let tiny_res = z.reciprocal_p::<P>() - Vf32::<S>::splat(EULERS_CONSTANT);
            res *= is_tiny.select(tiny_res, normal_res);
        } else {
            res *= normal_res;
        }

        reflected.select(-pi / res, z_int.select(fact_res, res))
    }

    #[inline(always)]
    fn lgamma<P: Policy>(mut z: Self::Vf) -> Self::Vf {
        let one = Vf32::<S>::one();
        let zero = Vf32::<S>::zero();

        let reflect = z.lt(zero);

        let mut t = one;

        if P::POLICY.avoid_branching || reflect.any() {
            t = reflect.select(<Self as SimdVectorizedMathInternal<S>>::sin_pix::<P>(z).abs(), one);
            z = z.conditional_neg(reflect);
        }

        let gh = Vf32::<S>::splat(LANCZOS_G - 0.5);

        let mut lanczos_sum = z.poly_rational_p::<P>(LANCZOS_P_EXPG_SCALED, LANCZOS_Q);

        // Full A
        let mut a = (z + gh).ln_p::<P>() - one;

        // Tiny
        if P::POLICY.precision >= PrecisionPolicy::Best {
            let is_not_tiny = z.ge(Vf32::<S>::splat_as(
                <Self as SimdVectorizedMathInternal<S>>::__SQRT_EPSILON,
            ));
            let tiny_res = z.reciprocal_p::<P>() - Vf32::<S>::splat(EULERS_CONSTANT);

            // shove the tiny result into the log down below
            lanczos_sum = is_not_tiny.select(lanczos_sum, tiny_res);
            // force multiplier to zero for tiny case, allowing the modified
            // lanczos sum and ln(t) to be combined for cheap
            a &= is_not_tiny.value();
        }

        // Full

        let b = z - Vf32::<S>::splat(0.5);
        let c = (lanczos_sum * t).ln_p::<P>();

        let mut res = a.mul_adde(b, c);

        let ln_pi = Vf32::<S>::splat(LN_PI);

        res = reflect.select(ln_pi - res, res);

        res
    }

    #[inline(always)]
    fn digamma<P: Policy>(mut x: Self::Vf) -> Self::Vf {
        let zero = Vf32::<S>::zero();
        let one = Vf32::<S>::one();
        let half = Vf32::<S>::splat(0.5);
        let pi = Vf32::<S>::splat(PI);

        let mut result = zero;

        let reflect = x.le(Vf32::<S>::neg_one());

        if reflect.any() {
            x = reflect.select(one - x, x);

            let mut rem = x - x.floor();

            rem = rem.conditional_sub(one, rem.gt(half));

            let (s, c) = (rem * pi).sin_cos_p::<P>();
            let refl_res = pi * c / s;

            result = reflect.select(refl_res, result);
        }

        let lim = Vf32::<S>::splat(
            0.5 * (10 + ((<Self as SimdVectorizedMathInternal<S>>::__DIGITS as i64 - 50) * 240) / 950) as f32,
        );

        // Rescale to use asymptotic expansion
        let mut is_small = x.lt(lim);
        while is_small.any() {
            result = result.conditional_sub(x.reciprocal_p::<P>(), is_small);
            x = x.conditional_add(one, is_small);
            is_small = x.lt(lim);
        }

        x -= one;

        let inv_x = x.reciprocal_p::<P>();

        let z = inv_x * inv_x;
        let a = x.ln_p::<P>() + (inv_x * half);

        let y = z.poly_p::<P>(&[
            0.083333333333333333333333333333333333333333333333333,
            -0.0083333333333333333333333333333333333333333333333333,
            0.003968253968253968253968253968253968253968253968254,
        ]);

        result += z.nmul_adde(y, a);

        result
    }

    #[inline(always)]
    fn beta<P: Policy>(a: Self::Vf, b: Self::Vf) -> Self::Vf {
        let zero = Vf32::<S>::zero();
        let one = Vf32::<S>::one();

        let is_valid = a.gt(zero) & b.gt(zero);

        if P::POLICY.check_overflow && !P::POLICY.avoid_branching {
            if is_valid.none() {
                return Vf32::<S>::nan();
            }
        }

        let c = a + b;

        // if a < b then swap
        let (a, b) = (a.max(b), a.min(b));

        let mut result = a.poly_rational_p::<P>(LANCZOS_P_EXPG_SCALED, LANCZOS_Q)
            * (b.poly_rational_p::<P>(LANCZOS_P_EXPG_SCALED, LANCZOS_Q)
                / c.poly_rational_p::<P>(LANCZOS_P_EXPG_SCALED, LANCZOS_Q));

        let gh = Vf32::<S>::splat(LANCZOS_G - 0.5);

        let agh = a + gh;
        let bgh = b + gh;
        let cgh = c + gh;

        let agh_d_cgh = agh / cgh;
        let bgh_d_cgh = bgh / cgh;
        let agh_p_bgh = agh * bgh;
        let cgh_p_cgh = cgh * cgh;

        let base = cgh
            .gt(Vf32::<S>::splat(1e10))
            .select(agh_d_cgh * bgh_d_cgh, agh_p_bgh / cgh_p_cgh);

        let denom = if P::POLICY.precision > PrecisionPolicy::Average {
            Vf32::<S>::splat(SQRT_E) / bgh.sqrt()
        } else {
            // bump up the precision a little to improve beta function accuracy
            Vf32::<S>::splat(SQRT_E) * bgh.invsqrt_p::<ExtraPrecision<P>>()
        };

        result *= agh_d_cgh.powf_p::<P>(a - Vf32::<S>::splat(0.5) - b) * (base.powf_p::<P>(b) * denom);

        if P::POLICY.check_overflow {
            result = is_valid.select(result, Vf32::<S>::nan());
        }

        result
    }
}

#[inline(always)]
fn pow2n_f<S: Simd>(n: Vf32<S>) -> Vf32<S> {
    let pow2_23 = Vf32::<S>::splat(8388608.0);
    let bias = Vf32::<S>::splat(127.0);

    (n + (bias + pow2_23)) << 23
}

#[inline(always)]
fn exp_f_internal<S: Simd, P: Policy>(x0: Vf32<S>, mode: ExpMode) -> Vf32<S> {
    let mut x = x0;
    let mut r;

    let max_x = match mode {
        ExpMode::Exp => 87.3,
        ExpMode::Exph | ExpMode::Expm1 => 89.0,
        ExpMode::Pow2 => 126.0,
        ExpMode::Pow10 => 37.9,
    };

    let mut z = if P::POLICY.precision == PrecisionPolicy::Worst {
        // https://stackoverflow.com/a/10792321 with a better 2^f fit

        // Compute t such that b^x = 2^t
        let t = match mode {
            ExpMode::Exp | ExpMode::Exph | ExpMode::Expm1 => x * Vf32::<S>::splat(LOG2_E),
            ExpMode::Pow2 => x,
            ExpMode::Pow10 => x * Vf32::<S>::splat(LOG10_2),
        };

        let fi = t.floor();
        let f = t - fi;

        // if the exponent exceeds this method's limitations, then it's far outside of the valid range for exp
        let i = unsafe { fi.to_int_fast() };

        // polynomial approximation of 2^f
        let cf = f.poly_p::<P>(&[1.0, 0.695556856, 0.226173572, 0.0781455737]);

        // scale 2^f by 2^i
        let ci = Vi32::<S>::from_bits(cf.into_bits()) + (i << 23);

        let z = Vf32::<S>::from_bits(ci.into_bits());

        match mode {
            ExpMode::Exph => z * Vf32::<S>::splat(0.5),
            ExpMode::Expm1 => z - Vf32::<S>::one(),
            _ => z,
        }
    } else {
        match mode {
            ExpMode::Exp | ExpMode::Exph | ExpMode::Expm1 => {
                let ln2f_hi = Vf32::<S>::splat(0.693359375);
                let ln2f_lo = Vf32::<S>::splat(-2.12194440e-4);

                r = (x0 * Vf32::<S>::splat(LOG2_E)).round();

                x = r.nmul_adde(ln2f_hi, x); // x -= r * ln2f_hi;
                x = r.nmul_adde(ln2f_lo, x); // x -= r * ln2f_lo;

                if mode == ExpMode::Exph {
                    r -= Vf32::<S>::one();
                }
            }
            ExpMode::Pow2 => {
                r = x0.round();

                x -= r;
                x *= Vf32::<S>::splat(LN_2);
            }
            ExpMode::Pow10 => {
                let log10_2_hi = Vf32::<S>::splat(0.301025391); // log10(2) in two parts
                let log10_2_lo = Vf32::<S>::splat(4.60503907E-6);

                r = (x0 * Vf32::<S>::splat(LN_10 * LOG2_E)).round();

                x = r.nmul_adde(log10_2_hi, x); // x -= r * log10_2_hi;
                x = r.nmul_adde(log10_2_lo, x); // x -= r * log10_2_lo;
                x *= Vf32::<S>::splat(LN_10);
            }
        }

        let mut z = x
            .poly_p::<P>(&[1.0 / 2.0, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0, 1.0 / 720.0, 1.0 / 5040.0])
            .mul_adde(x * x, x);

        let n2 = pow2n_f::<S>(r);

        match mode {
            ExpMode::Expm1 => z.mul_adde(n2, n2 - Vf32::<S>::one()),
            _ => z.mul_adde(n2, n2), // (z + 1.0f) * n2
        }
    };

    if P::POLICY.check_overflow {
        let in_range = x0.abs().lt(Vf32::<S>::splat(max_x)) & x0.is_finite();

        if likely!(in_range.all()) {
            return z;
        }

        let underflow_value = match mode {
            ExpMode::Expm1 => Vf32::<S>::neg_one(),
            _ => Vf32::<S>::zero(),
        };

        r = x0.select_negative(underflow_value, Vf32::<S>::infinity());
        z = in_range.select(z, r);
        z = x0.is_nan().select(x0, z);
    }

    z
}

#[inline(always)]
fn asin_f_internal<S: Simd, P: Policy>(x: Vf32<S>, acos: bool) -> Vf32<S> {
    let xa = x.abs();

    let is_big = xa.gt(Vf32::<S>::splat(0.5));

    // TODO: Branch to avoid sqrt?
    let x1 = Vf32::<S>::splat(0.5) * (Vf32::<S>::one() - xa);
    let x2 = xa * xa;
    let x3 = is_big.select(x1, x2);
    let xb = x1.sqrt();
    let x4 = is_big.select(xb, xa);

    let z = x3
        .poly_p::<P>(&[
            1.6666752422E-1,
            7.4953002686E-2,
            4.5470025998E-2,
            2.4181311049E-2,
            4.2163199048E-2,
        ])
        .mul_adde(x3 * x4, x4);

    let z1 = z + z;

    if acos {
        let z1 = x.select_negative(Vf32::<S>::splat(PI) - z1, z1);
        let z2 = Vf32::<S>::splat(FRAC_PI_2) - z.combine_sign(x);

        is_big.select(z1, z2)
    } else {
        let z1 = Vf32::<S>::splat(FRAC_PI_2) - z1;

        is_big.select(z1, z).combine_sign(x)
    }
}

#[inline(always)]
fn fraction2<S: Simd>(x: Vf32<S>) -> Vf32<S> {
    // set exponent to 0 + bias
    (x & Vf32::<S>::splat(f32::from_bits(0x007FFFFF))) | Vf32::<S>::splat(f32::from_bits(0x3F000000))
}

#[inline(always)]
fn exponent<S: Simd>(x: Vf32<S>) -> Vi32<S> {
    // shift out sign, extract exp, subtract bias
    Vi32::<S>::from_bits((x.into_bits() << 1) >> 24) - Vi32::<S>::splat(0x7F)
}

#[inline(always)]
fn ln_f_internal<S: Simd, P: Policy>(x0: Vf32<S>, p1: bool) -> Vf32<S> {
    let ln2f_hi = Vf32::<S>::splat(0.693359375);
    let ln2f_lo = Vf32::<S>::splat(-2.12194440E-4);
    let one = Vf32::<S>::one();

    let x1 = if p1 { x0 + one } else { x0 };

    let mut x = fraction2::<S>(x1);
    let mut e = exponent::<S>(x1);

    let blend = x.gt(Vf32::<S>::splat(SQRT_2 * 0.5));

    x = x.conditional_add(x, !blend);
    e = e.conditional_add(Vi32::<S>::one(), blend);

    // TODO: Fix this cast when the type inference bug hits stable
    let fe = <Vf32<S> as SimdFromCast<S, Vi32<S>>>::from_cast(e);

    let xp1 = x - one;

    x = if p1 {
        // log(x+1). Avoid loss of precision when adding 1 and later subtracting 1 if exponent = 0
        e.eq(Vi32::<S>::zero()).select(x0, xp1)
    } else {
        // log(x). Expand around 1.0
        xp1
    };

    let x2 = x * x;
    let mut res = x.poly_p::<P>(&[
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
    ]) * x2;

    res = fe.mul_adde(ln2f_lo, res);
    res += x2.nmul_adde(Vf32::<S>::splat(0.5), x);
    res = fe.mul_adde(ln2f_hi, res);

    if !P::POLICY.check_overflow {
        return res;
    }

    let overflow = !x1.is_finite();
    let underflow = x1.lt(Vf32::<S>::splat(1.17549435e-38));

    if likely!((overflow | underflow).none()) {
        return res;
    }

    res = underflow.select(Vf32::<S>::nan(), res); // x1 < 0 gives NAN
    res = x1.is_zero_or_subnormal().select(Vf32::<S>::neg_infinity(), res); // x1 == 0 gives -INF
    res = overflow.select(x1, res); // INF or NAN goes through
    res = (x1.is_infinite() & x1.is_negative()).select(Vf32::<S>::nan(), res); // -INF gives NAN

    res
}

const LANCZOS_G: f32 = 1.428456135094165802001953125;

const LANCZOS_P: &[f32] = &[
    58.52061591769095910314047740215847630266,
    182.5248962595894264831189414768236280862,
    211.0971093028510041839168287718170827259,
    112.2526547883668146736465390902227161763,
    27.5192015197455403062503721613097825345,
    2.50662858515256974113978724717473206342,
];

const LANCZOS_Q: &[f32] = &[0.0, 24.0, 50.0, 35.0, 10.0, 1.0];

const LANCZOS_P_EXPG_SCALED: &[f32] = &[
    14.0261432874996476619570577285003839357,
    43.74732405540314316089531289293124360129,
    50.59547402616588964511581430025589038612,
    26.90456680562548195593733429204228910299,
    6.595765571169314946316366571954421695196,
    0.6007854010515290065101128585795542383721,
];
