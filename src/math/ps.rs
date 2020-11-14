use super::{common::*, *};

use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, LN_10, LN_2, LOG10_2, LOG10_E, LOG2_E, PI, SQRT_2};

impl<S: Simd> SimdVectorizedMathInternal<S> for f32
where
    <S as Simd>::Vf32: SimdFloatVector<S, Element = f32>,
{
    type Vf = <S as Simd>::Vf32;

    const __EPSILON: Self = f32::EPSILON;

    #[inline(always)]
    fn from_u32(x: u32) -> Self {
        x as f32
    }

    #[inline(always)]
    fn from_i32(x: i32) -> Self {
        x as f32
    }

    #[inline(always)]
    fn sin_cos(xx: Self::Vf) -> (Self::Vf, Self::Vf) {
        let dp1f = Vf32::<S>::splat(0.78515625 * 2.0);
        let dp2f = Vf32::<S>::splat(2.4187564849853515625E-4 * 2.0);
        let dp3f = Vf32::<S>::splat(3.77489497744594108E-8 * 2.0);
        let p0sinf = Vf32::<S>::splat(-1.6666654611E-1);
        let p1sinf = Vf32::<S>::splat(8.3321608736E-3);
        let p2sinf = Vf32::<S>::splat(-1.9515295891E-4);
        let p0cosf = Vf32::<S>::splat(4.166664568298827E-2);
        let p1cosf = Vf32::<S>::splat(-1.388731625493765E-3);
        let p2cosf = Vf32::<S>::splat(2.443315711809948E-5);

        let xa: Vf32<S> = xx.abs();

        let y: Vf32<S> = (xa * Vf32::<S>::splat(2.0 / PI)).round();
        let q: Vu32<S> = y.cast_to::<Vi32<S>>().into_bits(); // cast to signed (faster), then transmute to unsigned

        // Reduce by extended precision modular arithmetic
        // x = ((xa - y * DP1F) - y * DP2F) - y * DP3F;
        let x = y.nmul_add(dp3f, y.nmul_add(dp2f, y.nmul_add(dp1f, xa)));

        // Taylor expansion of sin and cos, valid for -pi/4 <= x <= pi/4
        let x2: Vf32<S> = x * x;
        let x3: Vf32<S> = x2 * x;
        let x4: Vf32<S> = x2 * x2;

        let mut s = poly_2(x2, x4, p0sinf, p1sinf, p2sinf).mul_add(x3, x);
        let mut c =
            poly_2(x2, x4, p0cosf, p1cosf, p2cosf).mul_add(x4, Vf32::<S>::splat(0.5).nmul_add(x2, Vf32::<S>::one()));

        // swap sin and cos if odd quadrant
        let swap = (q & Vu32::<S>::one()).ne(Vu32::<S>::zero());

        let overflow = q.gt(Vu32::<S>::splat(0x2000000)) & xa.is_finite().cast_to(); // q big if overflow

        s = overflow.select(Vf32::<S>::zero(), s);
        c = overflow.select(Vf32::<S>::one(), c);

        let sin1 = swap.select(c, s);
        let cos1 = swap.select(s, c);

        let signsin = Vf32::<S>::from_bits((q << 30) ^ xx.into_bits());
        let signcos = Vf32::<S>::from_bits(((q + Vu32::<S>::one()) & Vu32::<S>::splat(2)) << 30);

        // combine signs
        (sin1.combine_sign(signsin), cos1 ^ signcos)
    }

    #[inline(always)]
    fn sinh(x0: Self::Vf) -> Self::Vf {
        let x = x0.abs();

        let x_small = x.le(Vf32::<S>::one());

        let mut y1 = unsafe { Vf32::<S>::undefined() };
        let mut y2 = unsafe { Vf32::<S>::undefined() };

        // use bitmask directly to avoid two calls
        let bitmask = x_small.bitmask();

        // if any are small
        if bitmask.any() {
            let r0 = Vf32::<S>::splat(1.66667160211E-1);
            let r1 = Vf32::<S>::splat(8.33028376239E-3);
            let r2 = Vf32::<S>::splat(2.03721912945E-4);

            let x2 = x * x;
            y1 = poly_2(x2, x2 * x2, r0, r1, r2).mul_add(x2 * x, x);
        }

        // if not all are small
        if !bitmask.all() {
            y2 = x.exph();
            y2 -= Vf32::<S>::splat(0.25) / y2;
        }

        x_small.select(y1, y2).combine_sign(x0)
    }

    #[inline(always)]
    fn tanh(x0: Self::Vf) -> Self::Vf {
        let one = Vf32::<S>::one();

        let x = x0.abs();
        let x_small = x.le(Vf32::<S>::splat(0.625));

        let mut y1 = unsafe { Vf32::<S>::undefined() };
        let mut y2 = unsafe { Vf32::<S>::undefined() };

        // use bitmask directly to avoid two calls
        let bitmask = x_small.bitmask();

        // if any are small
        if bitmask.any() {
            let r0 = Vf32::<S>::splat(-3.33332819422E-1);
            let r1 = Vf32::<S>::splat(1.33314422036E-1);
            let r2 = Vf32::<S>::splat(-5.37397155531E-2);
            let r3 = Vf32::<S>::splat(2.06390887954E-2);
            let r4 = Vf32::<S>::splat(-5.70498872745E-3);

            let x2 = x * x;
            let x4 = x2 * x2;

            y1 = poly_4(x2, x4, x4 * x4, r0, r1, r2, r3, r4).mul_add(x2 * x, x);
        }

        // if not all are small
        if !bitmask.all() {
            y2 = (x + x).exp();
            y2 = (y2 - one) / (y2 + one); // originally (1 - 2/(y2 + 1)), but doing it this way avoids loading 2.0
        }

        let x_big = x.gt(Vf32::<S>::splat(44.4));

        y1 = x_small.select(y1, y2);
        y1 = x_big.select(one, y1);

        y1.combine_sign(x0)
    }

    #[inline(always)]
    fn asin(x: Self::Vf) -> Self::Vf {
        asin_f_internal::<S>(x, false)
    }

    #[inline(always)]
    fn acos(x: Self::Vf) -> Self::Vf {
        asin_f_internal::<S>(x, true)
    }

    #[inline(always)]
    fn atan(y: Self::Vf) -> Self::Vf {
        let p3atanf = Vf32::<S>::splat(8.05374449538E-2);
        let p2atanf = Vf32::<S>::splat(-1.38776856032E-1);
        let p1atanf = Vf32::<S>::splat(1.99777106478E-1);
        let p0atanf = Vf32::<S>::splat(-3.33329491539E-1);

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

        poly_3(z2, z2 * z2, p0atanf, p1atanf, p2atanf, p3atanf)
            .mul_add(z2 * z, z + s)
            .combine_sign(y)
    }

    #[inline(always)]
    fn atan2(y: Self::Vf, x: Self::Vf) -> Self::Vf {
        let p3atanf = Vf32::<S>::splat(8.05374449538E-2);
        let p2atanf = Vf32::<S>::splat(-1.38776856032E-1);
        let p1atanf = Vf32::<S>::splat(1.99777106478E-1);
        let p0atanf = Vf32::<S>::splat(-3.33329491539E-1);
        let neg_one = Vf32::<S>::neg_one();
        let zero = Vf32::<S>::zero();

        let x1 = x.abs();
        let y1 = y.abs();

        let swap_xy = y1.gt(x1);

        let mut x2 = swap_xy.select(y1, x1);
        let mut y2 = swap_xy.select(x1, y1);

        let both_infinite = (x.is_infinite() & y.is_infinite());

        if unlikely!(both_infinite.any()) {
            x2 = both_infinite.select(x2 & neg_one, x2); // get 1.0 with the sign of x
            y2 = both_infinite.select(y2 & neg_one, y2); // get 1.0 with the sign of y
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

        let mut re = poly_3(z2, z2 * z2, p0atanf, p1atanf, p2atanf, p3atanf).mul_add(z2 * z, z + s);

        re = swap_xy.select(Vf32::<S>::splat(FRAC_PI_2) - re, re);
        re = (x | y).eq(zero).select(zero, re); // atan2(0,+0) = 0 by convention
        re = x.is_negative().select(Vf32::<S>::splat(PI) - re, re); // also for x = -0.

        re
    }

    #[inline(always)]
    fn asinh(x0: Self::Vf) -> Self::Vf {
        let x = x0.abs();
        let x2 = x0 * x0;

        let x_small = x.le(Vf32::<S>::splat(0.51));
        let x_huge = x.gt(Vf32::<S>::splat(1e10));

        let mut y1 = unsafe { Vf32::<S>::undefined() };
        let mut y2 = unsafe { Vf32::<S>::undefined() };

        let bitmask = x_small.bitmask();

        if bitmask.any() {
            let r0 = Vf32::<S>::splat(-1.6666288134E-1);
            let r1 = Vf32::<S>::splat(7.4847586088E-2);
            let r2 = Vf32::<S>::splat(-4.2699340972E-2);
            let r3 = Vf32::<S>::splat(2.0122003309E-2);

            y1 = poly_3(x2, x2 * x2, r0, r1, r2, r3).mul_add(x2 * x, x);
        }

        if !bitmask.all() {
            y2 = ((x2 + Vf32::<S>::one()).sqrt() + x).ln();

            if unlikely!(x_huge.any()) {
                y2 = x_huge.select(x.ln() + Vf32::<S>::splat(LN_2), y2);
            }
        }

        x_small.select(y1, y2).combine_sign(x0)
    }

    #[inline(always)]
    fn acosh(x0: Self::Vf) -> Self::Vf {
        let one = Vf32::<S>::one();

        let x1 = x0 - one;

        let undef = x0.lt(one); // result is NAN
        let x_small = x1.lt(Vf32::<S>::splat(0.49)); // use Pade approximation if abs(x-1) < 0.5
        let x_huge = x1.gt(Vf32::<S>::splat(1e10));

        let mut y1 = unsafe { Vf32::<S>::undefined() };
        let mut y2 = unsafe { Vf32::<S>::undefined() };

        let bitmask = x_small.bitmask();

        // if any are small
        if bitmask.any() {
            let r0 = Vf32::<S>::splat(1.4142135263E0);
            let r1 = Vf32::<S>::splat(-1.1784741703E-1);
            let r2 = Vf32::<S>::splat(2.6454905019E-2);
            let r3 = Vf32::<S>::splat(-7.5272886713E-3);
            let r4 = Vf32::<S>::splat(1.7596881071E-3);

            let x2 = x1 * x1;
            let x4 = x2 * x2;
            y1 = x1.sqrt() * poly_4(x1, x2, x4, r0, r1, r2, r3, r4);
            y1 = undef.select(Vf32::<S>::nan(), y1);
        }

        // if not all are small
        if !bitmask.all() {
            y2 = (x0.mul_sub(x0, one).sqrt() + x0).ln();

            if x_huge.any() {
                y2 = x_huge.select(x0.ln() + Vf32::<S>::splat(LN_2), y2);
            }
        }

        x_small.select(y1, y2)
    }

    #[inline(always)]
    fn atanh(x0: Self::Vf) -> Self::Vf {
        let x = x0.abs();

        let x_small = x.lt(Vf32::<S>::splat(0.5));

        let mut y1 = unsafe { Vf32::<S>::undefined() };
        let mut y2 = unsafe { Vf32::<S>::undefined() };

        let bitmask = x_small.bitmask();

        if bitmask.any() {
            let r0 = Vf32::<S>::splat(3.33337300303E-1);
            let r1 = Vf32::<S>::splat(1.99782164500E-1);
            let r2 = Vf32::<S>::splat(1.46691431730E-1);
            let r3 = Vf32::<S>::splat(8.24370301058E-2);
            let r4 = Vf32::<S>::splat(1.81740078349E-1);

            let x2 = x * x;
            let x4 = x2 * x2;
            let x8 = x4 * x4;

            y1 = poly_4(x2, x4, x8, r0, r1, r2, r3, r4).mul_add(x2 * x, x);
        }

        if !bitmask.all() {
            let one = Vf32::<S>::one();

            y2 = ((one + x) / (one - x)).ln() * Vf32::<S>::splat(0.5);

            let y3 = x.eq(one).select(Vf32::<S>::infinity(), Vf32::<S>::nan());
            y2 = x.ge(one).select(y3, y2);
        }

        x_small.select(y1, y2).combine_sign(x0)
    }

    #[inline(always)]
    fn exp(x: Self::Vf) -> Self::Vf {
        exp_f_internal::<S>(x, ExpMode::Exp)
    }

    #[inline(always)]
    fn exph(x: Self::Vf) -> Self::Vf {
        exp_f_internal::<S>(x, ExpMode::Exph)
    }

    #[inline(always)]
    fn exp2(x: Self::Vf) -> Self::Vf {
        exp_f_internal::<S>(x, ExpMode::Pow2)
    }

    #[inline(always)]
    fn exp10(x: Self::Vf) -> Self::Vf {
        exp_f_internal::<S>(x, ExpMode::Pow10)
    }

    #[inline(always)]
    fn exp_m1(x: Self::Vf) -> Self::Vf {
        exp_f_internal::<S>(x, ExpMode::Expm1)
    }

    #[inline(always)]
    fn cbrt(x: Self::Vf) -> Self::Vf {
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

        // "fast" integer division `hx / 3`
        //let hx_j = <Vu64<S> as SimdCastFrom<S, Vu32<S>>>::from_cast(hx) * Vu64::<S>::splat(0xAAAAAAAB);
        //let hx_3 = <Vu32<S> as SimdCastFrom<S, Vu64<S>>>::from_cast(hx_j >> 33);

        hx = hx.div_const(3) + b;

        ui &= Vu32::<S>::splat(0x80000000);
        ui |= hx;

        let mut t = Vf32::<S>::from_bits(ui);

        let two = Vf32::<S>::splat(2.0);

        // couple iterations of Newton's method
        // This isn't perfect, as it's only limited to single-precision,
        // but the fused multiply-adds helps
        for _ in 0..2 {
            let t3 = t * t * t;
            t *= two.mul_add(x, t3) / two.mul_add(t3, x); // try to use extended precision where possible
        }

        // cbrt(NaN,INF,+-0) is itself
        (hx0.gt(Vu32::<S>::splat(0x7f800000)) | hx0.eq(Vu32::<S>::zero())).select(x, t)
    }

    #[inline(always)]
    fn powf(x0: Self::Vf, y: Self::Vf) -> Self::Vf {
        // define constants
        let ln2f_hi = Vf32::<S>::splat(0.693359375); // log(2), split in two for extended precision
        let ln2f_lo = Vf32::<S>::splat(-2.12194440e-4);
        let log2e = Vf32::<S>::splat(LOG2_E); // 1/ln(2)
        let ln2 = Vf32::<S>::splat(LN_2);

        // coefficients for logarithm expansion
        let p0logf = Vf32::<S>::splat(3.3333331174E-1);
        let p1logf = Vf32::<S>::splat(-2.4999993993E-1);
        let p2logf = Vf32::<S>::splat(2.0000714765E-1);
        let p3logf = Vf32::<S>::splat(-1.6668057665E-1);
        let p4logf = Vf32::<S>::splat(1.4249322787E-1);
        let p5logf = Vf32::<S>::splat(-1.2420140846E-1);
        let p6logf = Vf32::<S>::splat(1.1676998740E-1);
        let p7logf = Vf32::<S>::splat(-1.1514610310E-1);
        let p8logf = Vf32::<S>::splat(7.0376836292E-2);

        // coefficients for Taylor expansion of exp
        let p2expf = Vf32::<S>::splat(1.0 / 2.0);
        let p3expf = Vf32::<S>::splat(1.0 / 6.0);
        let p4expf = Vf32::<S>::splat(1.0 / 24.0);
        let p5expf = Vf32::<S>::splat(1.0 / 120.0);
        let p6expf = Vf32::<S>::splat(1.0 / 720.0);
        let p7expf = Vf32::<S>::splat(1.0 / 5040.0);

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
        let x4 = x2 * x2;
        let mut lg1 = poly_8(
            x,
            x2,
            x4,
            x4 * x4,
            p0logf,
            p1logf,
            p2logf,
            p3logf,
            p4logf,
            p5logf,
            p6logf,
            p7logf,
            p8logf,
        );
        lg1 *= x2 * x;

        let ef = <Vf32<S> as SimdCastFrom<S, Vi32<S>>>::from_cast(exponent::<S>(x1)) + (blend.value() & one);

        // multiply exponent by y, nearest integer e1 goes into exponent of result, remainder yr is added to log
        let e1 = (ef * y).round();
        let yr = ef.mul_sub(y, e1); // calculate remainder yr. precision very important here

        // add initial terms to expansion
        let lg = half.nmul_add(x2, x) + lg1; // lg = (x - 0.5f * x2) + lg1;

        // calculate rounding errors in lg
        // rounding error in multiplication 0.5*x*x
        let x2err = (half * x).mul_sub(x, half * x2);

        // rounding error in additions and subtractions
        let lgerr = half.mul_add(x2, lg - x) - lg1; // lgerr = ((lg - x) + 0.5f * x2) - lg1;

        // extract something for the exponent
        let e2 = (lg * y * log2e).round();

        // subtract this from lg, with extra precision
        let mut v = e2.nmul_add(ln2f_lo, lg.mul_sub(y, e2 * ln2f_hi));

        // correct for previous rounding errors
        v -= (lgerr + x2err).mul_sub(y, yr * ln2);

        // extract something for the exponent if possible
        let mut x = v;
        let e3 = (x * log2e).round();

        // high precision multiplication not needed here because abs(e3) <= 1
        x = e3.nmul_add(ln2, x); // x -= e3 * float(VM_LN2);

        let x2 = x * x;
        let x4 = x2 * x2;
        let z = poly_5(x, x2, x4, p2expf, p3expf, p4expf, p5expf, p6expf, p7expf).mul_add(x2, x + one);

        // contributions to exponent
        let ee = e1 + e2 + e3;
        let ei = ee.cast_to::<Vi32<S>>();

        // biased exponent of result:
        let ej = ei + Vi32::<S>::from_bits(z.into_bits()) >> 23;

        // check exponent for overflow and underflow
        let overflow = Vf32::<S>::from_cast_mask(ej.ge(Vi32::<S>::splat(0x0FF))) | ee.gt(Vf32::<S>::splat(300.0));
        let underflow = Vf32::<S>::from_cast_mask(ej.le(Vi32::<S>::splat(0x000))) | ee.lt(Vf32::<S>::splat(-300.0));

        // add exponent by signed integer addition
        let mut z = Vf32::<S>::from_bits((Vi32::<S>::from_bits(z.into_bits()) + (ei << 23)).into_bits());

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
    fn ln(x: Self::Vf) -> Self::Vf {
        ln_f_internal::<S>(x, false)
    }

    #[inline(always)]
    fn ln_1p(x: Self::Vf) -> Self::Vf {
        ln_f_internal::<S>(x, true)
    }

    #[inline(always)]
    fn log2(x: Self::Vf) -> Self::Vf {
        x.ln() * Vf32::<S>::splat(LOG2_E)
    }

    #[inline(always)]
    fn log10(x: Self::Vf) -> Self::Vf {
        x.ln() * Vf32::<S>::splat(LOG10_E)
    }

    #[inline(always)]
    fn erf(x: Self::Vf) -> Self::Vf {
        x * (x * x).poly(&[
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
    fn erfinv(y: Self::Vf) -> Self::Vf {
        /*
            Approximating the erfinv function, Mike Giles
            https://people.maths.ox.ac.uk/gilesm/files/gems_erfinv.pdf
        */
        let one = Vf32::<S>::one();

        let a = y.abs();

        let w = -a.nmul_add(a, one).ln();

        let mut p0 = (w - Vf32::<S>::splat(2.5)).poly(&[
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
            let mut p1 = (w.sqrt() - Vf32::<S>::splat(3.0)).poly(&[
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

            p1 = a.eq(one).select(Vf32::<S>::infinity(), p1); // erfinv(x == 1) = inf
            p1 = a.gt(one).select(Vf32::<S>::nan(), p1); // erfinv(x > 1) = NaN

            p0 = w_big.select(p1, p0);
        }

        p0 * y
    }

    #[inline(always)]
    fn next_float(x: Self::Vf) -> Self::Vf {
        let i1 = Vu32::<S>::one();

        let v = x.eq(Vf32::<S>::neg_zero()).select(Vf32::<S>::zero(), x);

        let bits = v.into_bits();
        x.eq(Vf32::<S>::infinity()).select(
            x,
            Vf32::<S>::from_bits(v.ge(Vf32::<S>::zero()).select(bits + i1, bits - i1)),
        )
    }

    #[inline(always)]
    fn prev_float(x: Self::Vf) -> Self::Vf {
        let i1 = Vu32::<S>::one();

        let v = x.eq(Vf32::<S>::zero()).select(Vf32::<S>::neg_zero(), x);

        let bits = v.into_bits();
        x.eq(Vf32::<S>::neg_infinity()).select(
            x,
            Vf32::<S>::from_bits(v.gt(Vf32::<S>::zero()).select(bits - i1, bits + i1)),
        )
    }
}

#[inline(always)]
fn pow2n_f<S: Simd>(n: Vf32<S>) -> Vf32<S> {
    let pow2_23 = Vf32::<S>::splat(8388608.0);
    let bias = Vf32::<S>::splat(127.0);

    (n + (bias + pow2_23)) << 23
}

#[inline(always)]
fn exp_f_internal<S: Simd>(x0: Vf32<S>, mode: ExpMode) -> Vf32<S> {
    let p0expf = Vf32::<S>::splat(1.0 / 2.0);
    let p1expf = Vf32::<S>::splat(1.0 / 6.0);
    let p2expf = Vf32::<S>::splat(1.0 / 24.0);
    let p3expf = Vf32::<S>::splat(1.0 / 120.0);
    let p4expf = Vf32::<S>::splat(1.0 / 720.0);
    let p5expf = Vf32::<S>::splat(1.0 / 5040.0);

    let mut x = x0;
    let mut r;

    let max_x;

    match mode {
        ExpMode::Exp | ExpMode::Exph | ExpMode::Expm1 => {
            max_x = if mode == ExpMode::Exp { 87.3 } else { 89.0 };

            let ln2f_hi = Vf32::<S>::splat(0.693359375);
            let ln2f_lo = Vf32::<S>::splat(-2.12194440e-4);

            r = (x0 * Vf32::<S>::splat(LOG2_E)).round();

            x = r.nmul_add(ln2f_hi, x); // x -= r * ln2f_hi;
            x = r.nmul_add(ln2f_lo, x); // x -= r * ln2f_lo;

            if mode == ExpMode::Exph {
                r -= Vf32::<S>::one();
            }
        }
        ExpMode::Pow2 => {
            max_x = 126.0;

            r = x0.round();

            x -= r;
            x *= Vf32::<S>::splat(LN_2);
        }
        ExpMode::Pow10 => {
            max_x = 37.9;

            let log10_2_hi = Vf32::<S>::splat(0.301025391); // log10(2) in two parts
            let log10_2_lo = Vf32::<S>::splat(4.60503907E-6);

            r = (x0 * Vf32::<S>::splat(LN_10 * LOG2_E)).round();

            x = r.nmul_add(log10_2_hi, x); // x -= r * log10_2_hi;
            x = r.nmul_add(log10_2_lo, x); // x -= r * log10_2_lo;
            x *= Vf32::<S>::splat(LN_10);
        }
    }

    let x2 = x * x;
    let mut z = poly_5(x, x2, x2 * x2, p0expf, p1expf, p2expf, p3expf, p4expf, p5expf).mul_add(x2, x);

    let n2 = pow2n_f::<S>(r);

    if mode == ExpMode::Expm1 {
        z = z.mul_add(n2, n2 - Vf32::<S>::one());
    } else {
        z = z.mul_add(n2, n2); // (z + 1.0f) * n2
    }

    let in_range = x0.abs().lt(Vf32::<S>::splat(max_x)) & x0.is_finite().cast_to();

    if likely!(in_range.all()) {
        return z;
    }

    let sign_bit_mask = (x0 & Vf32::<S>::neg_zero()).into_bits().ne(Vu32::<S>::zero());
    let is_nan = x0.is_nan();

    let underflow_value = if mode == ExpMode::Expm1 {
        Vf32::<S>::neg_one()
    } else {
        Vf32::<S>::zero()
    };

    r = sign_bit_mask.select(underflow_value, Vf32::<S>::infinity());
    z = in_range.select(z, r);
    z = is_nan.select(x0, z);

    z
}

#[inline(always)]
fn asin_f_internal<S: Simd>(x: Vf32<S>, acos: bool) -> Vf32<S> {
    let p4asinf = Vf32::<S>::splat(4.2163199048E-2);
    let p3asinf = Vf32::<S>::splat(2.4181311049E-2);
    let p2asinf = Vf32::<S>::splat(4.5470025998E-2);
    let p1asinf = Vf32::<S>::splat(7.4953002686E-2);
    let p0asinf = Vf32::<S>::splat(1.6666752422E-1);

    let xa = x.abs();

    let is_big = xa.gt(Vf32::<S>::splat(0.5));

    let x1 = Vf32::<S>::splat(0.5) * (Vf32::<S>::one() - xa);
    let x2 = xa * xa;
    let x3 = is_big.select(x1, x2);
    let xb = x1.sqrt();
    let x4 = is_big.select(xb, xa);

    let xx = x3;
    let xx2 = xx * xx;
    let xx4 = xx2 * xx2;

    let z = poly_4(xx, xx2, xx4, p0asinf, p1asinf, p2asinf, p3asinf, p4asinf).mul_add(x3 * x4, x4);

    let z1 = z + z;

    if acos {
        let z1 = x.is_positive().select(Vf32::<S>::splat(PI) - z1, z1);
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
fn ln_f_internal<S: Simd>(x0: Vf32<S>, p1: bool) -> Vf32<S> {
    let ln2f_hi = Vf32::<S>::splat(0.693359375);
    let ln2f_lo = Vf32::<S>::splat(-2.12194440E-4);
    let p0logf = Vf32::<S>::splat(3.3333331174E-1);
    let p1logf = Vf32::<S>::splat(-2.4999993993E-1);
    let p2logf = Vf32::<S>::splat(2.0000714765E-1);
    let p3logf = Vf32::<S>::splat(-1.6668057665E-1);
    let p4logf = Vf32::<S>::splat(1.4249322787E-1);
    let p5logf = Vf32::<S>::splat(-1.2420140846E-1);
    let p6logf = Vf32::<S>::splat(1.1676998740E-1);
    let p7logf = Vf32::<S>::splat(-1.1514610310E-1);
    let p8logf = Vf32::<S>::splat(7.0376836292E-2);
    let one = Vf32::<S>::one();

    let x1 = if p1 { x0 + one } else { x0 };

    let mut x = fraction2::<S>(x1);
    let mut e = exponent::<S>(x1);

    let blend = x.gt(Vf32::<S>::splat(SQRT_2 * 0.5));

    x += !blend.value() & x; // conditional addition
    e += Vi32::<S>::from_bits(blend.value().into_bits() & Vu32::<S>::one()); // conditional (signed) addition

    // TODO: Fix this cast when the type inference bug hits stable
    let fe = <Vf32<S> as SimdCastFrom<S, Vi32<S>>>::from_cast(e);

    let xp1 = x - one;

    x = if p1 {
        // log(x+1). Avoid loss of precision when adding 1 and later subtracting 1 if exponent = 0
        e.eq(Vi32::<S>::zero()).select(x0, xp1)
    } else {
        // log(x). Expand around 1.0
        xp1
    };

    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x2 * x2;

    let mut res = poly_8(
        x,
        x2,
        x4,
        x4 * x4,
        p0logf,
        p1logf,
        p2logf,
        p3logf,
        p4logf,
        p5logf,
        p6logf,
        p7logf,
        p8logf,
    ) * x3;

    res = fe.mul_add(ln2f_lo, res);
    res += x2.nmul_add(Vf32::<S>::splat(0.5), x);
    res = fe.mul_add(ln2f_hi, res);

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
