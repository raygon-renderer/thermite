use super::{common::*, *};

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, LN_10, LN_2, LOG10_2, LOG10_E, LOG2_E, PI, SQRT_2};

impl<S: Simd> SimdVectorizedMathInternal<S> for f64
where
    <S as Simd>::Vf64: SimdFloatVector<S, Element = f64>,
{
    type Vf = <S as Simd>::Vf64;

    const __EPSILON: Self = f64::EPSILON;

    #[inline(always)]
    fn from_u32(x: u32) -> Self {
        x as f64
    }

    #[inline(always)]
    fn sin_cos(xx: Self::Vf) -> (Self::Vf, Self::Vf) {
        let dp1 = Vf64::<S>::splat(7.853981554508209228515625E-1 * 2.0);
        let dp2 = Vf64::<S>::splat(7.94662735614792836714E-9 * 2.0);
        let dp3 = Vf64::<S>::splat(3.06161699786838294307E-17 * 2.0);
        let p0sin = Vf64::<S>::splat(-1.66666666666666307295E-1);
        let p1sin = Vf64::<S>::splat(8.33333333332211858878E-3);
        let p2sin = Vf64::<S>::splat(-1.98412698295895385996E-4);
        let p3sin = Vf64::<S>::splat(2.75573136213857245213E-6);
        let p4sin = Vf64::<S>::splat(-2.50507477628578072866E-8);
        let p5sin = Vf64::<S>::splat(1.58962301576546568060E-10);
        let p0cos = Vf64::<S>::splat(4.16666666666665929218E-2);
        let p1cos = Vf64::<S>::splat(-1.38888888888730564116E-3);
        let p2cos = Vf64::<S>::splat(2.48015872888517045348E-5);
        let p3cos = Vf64::<S>::splat(-2.75573141792967388112E-7);
        let p4cos = Vf64::<S>::splat(2.08757008419747316778E-9);
        let p5cos = Vf64::<S>::splat(-1.13585365213876817300E-11);
        let zero = Vf64::<S>::zero();
        let one = Vf64::<S>::one();

        let xa = xx.abs();

        let y = (xa * Vf64::<S>::splat(2.0 / PI)).round();
        let q = unsafe { y.to_uint_fast() };

        // Reduce by extended precision modular arithmetic
        // x = ((xa - y * DP1F) - y * DP2F) - y * DP3F;
        let x = y.nmul_add(dp3, y.nmul_add(dp2, y.nmul_add(dp1, xa)));

        // Taylor expansion of sin and cos, valid for -pi/4 <= x <= pi/4
        let x2 = x * x;
        let x3 = x2 * x;
        let x4 = x2 * x2;
        let x8 = x4 * x4;

        let mut s = poly_5(x2, x4, x8, p0sin, p1sin, p2sin, p3sin, p4sin, p5sin);
        let mut c = poly_5(x2, x4, x8, p0cos, p1cos, p2cos, p3cos, p4cos, p5cos);

        s = s.mul_add(x2 * x, x); // s = x + (x * x2) * s;
        c = c.mul_add(x4, x2.nmul_add(Vf64::<S>::splat(0.5), one)); // c = 1.0 - x2 * 0.5 + (x2 * x2) * c;

        // swap sin and cos if odd quadrant
        let swap = (q & Vu64::<S>::one()).ne(Vu64::<S>::zero());

        let overflow = y.gt(Vf64::<S>::splat((1u64 << 52) as f64 - 1.0)) & xa.is_finite();

        let s = overflow.select(zero, s);
        let c = overflow.select(one, c);

        let sin1 = swap.select(c, s);
        let cos1 = swap.select(s, c);

        let signsin = Vf64::<S>::from_bits((q << 62) ^ xx.into_bits());
        let signcos = Vf64::<S>::from_bits(((q + Vu64::<S>::one()) & Vu64::<S>::splat(2)) << 62);

        // combine signs
        (sin1.combine_sign(signsin), cos1 ^ signcos)
    }

    #[inline(always)]
    fn asin(x: Self::Vf) -> Self::Vf {
        asin_internal::<S>(x, false)
    }

    #[inline(always)]
    fn acos(x: Self::Vf) -> Self::Vf {
        asin_internal::<S>(x, true)
    }

    #[inline(always)]
    fn atan(x: Self::Vf) -> Self::Vf {
        atan_internal::<S>(x, unsafe { Vf64::<S>::undefined() }, false)
    }
    #[inline(always)]
    fn atan2(y: Self::Vf, x: Self::Vf) -> Self::Vf {
        atan_internal::<S>(y, x, true)
    }

    #[inline(always)]
    fn sinh(x0: Self::Vf) -> Self::Vf {
        let one = Vf64::<S>::one();

        let p0 = Vf64::<S>::splat(-3.51754964808151394800E5);
        let p1 = Vf64::<S>::splat(-1.15614435765005216044E4);
        let p2 = Vf64::<S>::splat(-1.63725857525983828727E2);
        let p3 = Vf64::<S>::splat(-7.89474443963537015605E-1);
        let q0 = Vf64::<S>::splat(-2.11052978884890840399E6);
        let q1 = Vf64::<S>::splat(3.61578279834431989373E4);
        let q2 = Vf64::<S>::splat(-2.77711081420602794433E2);
        let q3 = one;

        let x = x0.abs();

        let x_small = x.le(one);

        let mut y1 = unsafe { Vf64::<S>::undefined() };
        let mut y2 = unsafe { Vf64::<S>::undefined() };

        // use bitmask directly to avoid two calls
        let bitmask = x_small.bitmask();

        // if any are small
        if bitmask.any() {
            let x2 = x * x;
            let x4 = x2 * x2;

            y1 = poly_3(x2, x4, p0, p1, p2, p3) / poly_3(x2, x4, q0, q1, q2, q3);
            y1 = y1.mul_add(x * x2, x);
        }

        // if not all are small
        if !bitmask.all() {
            y2 = x.exph();
            y2 -= Vf64::<S>::splat(0.25) / y2;
        }

        x_small.select(y1, y2).combine_sign(x0)
    }

    #[inline(always)]
    fn tanh(x0: Self::Vf) -> Self::Vf {
        let one = Vf64::<S>::one();

        let p0 = Vf64::<S>::splat(-1.61468768441708447952E3);
        let p1 = Vf64::<S>::splat(-9.92877231001918586564E1);
        let p2 = Vf64::<S>::splat(-9.64399179425052238628E-1);
        let q0 = Vf64::<S>::splat(4.84406305325125486048E3);
        let q1 = Vf64::<S>::splat(2.23548839060100448583E3);
        let q2 = Vf64::<S>::splat(1.12811678491632931402E2);
        let q3 = one;

        let x = x0.abs();

        let x_small = x.le(Vf64::<S>::splat(0.625));

        let mut y1 = unsafe { Vf64::<S>::undefined() };
        let mut y2 = unsafe { Vf64::<S>::undefined() };

        // use bitmask directly to avoid two calls
        let bitmask = x_small.bitmask();

        // if any are small
        if bitmask.any() {
            let x2 = x * x;
            let x4 = x2 * x2;

            y1 = poly_2(x2, x4, p0, p1, p2) / poly_3(x2, x4, q0, q1, q2, q3);
            y1 = y1.mul_add(x2 * x, x);
        }

        // if not all are small
        if !bitmask.all() {
            y2 = (x + x).exp();
            y2 = (y2 - one) / (y2 + one); // originally (1 - 2/(y2 + 1)), but doing it this way avoids loading 2.0
        }

        let x_big = x.gt(Vf64::<S>::splat(350.0));

        y1 = x_small.select(y1, y2);
        y1 = x_big.select(one, y1);

        y1.combine_sign(x0)
    }

    #[inline(always)]
    fn asinh(x0: Self::Vf) -> Self::Vf {
        let one = Vf64::<S>::one();

        let p0 = Vf64::<S>::splat(-5.56682227230859640450E0);
        let p1 = Vf64::<S>::splat(-9.09030533308377316566E0);
        let p2 = Vf64::<S>::splat(-4.37390226194356683570E0);
        let p3 = Vf64::<S>::splat(-5.91750212056387121207E-1);
        let p4 = Vf64::<S>::splat(-4.33231683752342103572E-3);
        let q0 = Vf64::<S>::splat(3.34009336338516356383E1);
        let q1 = Vf64::<S>::splat(6.95722521337257608734E1);
        let q2 = Vf64::<S>::splat(4.86042483805291788324E1);
        let q3 = Vf64::<S>::splat(1.28757002067426453537E1);
        let q4 = one;

        let x = x0.abs();
        let x2 = x0 * x0;

        let x_small = x.le(Vf64::<S>::splat(0.533));
        let x_huge = x.gt(Vf64::<S>::splat(1e20));

        let mut y1 = unsafe { Vf64::<S>::undefined() };
        let mut y2 = unsafe { Vf64::<S>::undefined() };

        let bitmask = x_small.bitmask();

        if bitmask.any() {
            let x4 = x2 * x2;
            let x8 = x4 * x4;

            y1 = poly_4(x2, x4, x8, p0, p1, p2, p3, p4) / poly_4(x2, x4, x8, q0, q1, q2, q3, q4);
            y1 = y1.mul_add(x2 * x, x);
        }

        if !bitmask.all() {
            y2 = ((x2 + one).sqrt() + x).ln();

            if unlikely!(x_huge.any()) {
                y2 = x_huge.select(x.ln() + Vf64::<S>::splat(LN_2), y2);
            }
        }

        x_small.select(y1, y2).combine_sign(x0)
    }

    #[inline(always)]
    fn acosh(x0: Self::Vf) -> Self::Vf {
        let one = Vf64::<S>::one();

        let p0 = Vf64::<S>::splat(1.10855947270161294369E5);
        let p1 = Vf64::<S>::splat(1.08102874834699867335E5);
        let p2 = Vf64::<S>::splat(3.43989375926195455866E4);
        let p3 = Vf64::<S>::splat(3.94726656571334401102E3);
        let p4 = Vf64::<S>::splat(1.18801130533544501356E2);
        let q0 = Vf64::<S>::splat(7.83869920495893927727E4);
        let q1 = Vf64::<S>::splat(8.29725251988426222434E4);
        let q2 = Vf64::<S>::splat(2.97683430363289370382E4);
        let q3 = Vf64::<S>::splat(4.15352677227719831579E3);
        let q4 = Vf64::<S>::splat(1.86145380837903397292E2);
        let q5 = one;

        let x1 = x0 - one;

        let is_undef = x0.lt(one);
        let x_small = x1.lt(Vf64::<S>::splat(0.49));
        let x_huge = x1.gt(Vf64::<S>::splat(1e20));

        let mut y1 = unsafe { Vf64::<S>::undefined() };
        let mut y2 = unsafe { Vf64::<S>::undefined() };

        let bitmask = x_small.bitmask();

        if bitmask.any() {
            let x2 = x1 * x1;
            let x4 = x2 * x2;

            y1 = x1.sqrt() * (poly_4(x1, x2, x4, p0, p1, p2, p3, p4) / poly_5(x1, x2, x4, q0, q1, q2, q3, q4, q5));
            y1 = is_undef.select(Vf64::<S>::nan(), y1);
        }

        if !bitmask.all() {
            y2 = (x0.mul_sub(x0, one).sqrt() + x0).ln();

            if unlikely!(x_huge.any()) {
                y2 = x_huge.select(x0.ln() + Vf64::<S>::splat(LN_2), y2);
            }
        }

        x_small.select(y1, y2)
    }

    #[inline(always)]
    fn atanh(x0: Self::Vf) -> Self::Vf {
        let one = Vf64::<S>::one();
        let half = Vf64::<S>::splat(0.5);

        let p0 = Vf64::<S>::splat(-3.09092539379866942570E1);
        let p1 = Vf64::<S>::splat(6.54566728676544377376E1);
        let p2 = Vf64::<S>::splat(-4.61252884198732692637E1);
        let p3 = Vf64::<S>::splat(1.20426861384072379242E1);
        let p4 = Vf64::<S>::splat(-8.54074331929669305196E-1);
        let q0 = Vf64::<S>::splat(-9.27277618139601130017E1);
        let q1 = Vf64::<S>::splat(2.52006675691344555838E2);
        let q2 = Vf64::<S>::splat(-2.49839401325893582852E2);
        let q3 = Vf64::<S>::splat(1.08938092147140262656E2);
        let q4 = Vf64::<S>::splat(-1.95638849376911654834E1);
        let q5 = one;

        let x = x0.abs();

        let x_small = x.lt(half);

        let mut y1 = unsafe { Vf64::<S>::undefined() };
        let mut y2 = unsafe { Vf64::<S>::undefined() };

        let bitmask = x_small.bitmask();

        if bitmask.any() {
            let x2 = x * x;
            let x4 = x2 * x2;
            let x8 = x4 * x4;

            y1 = poly_4(x2, x4, x8, p0, p1, p2, p3, p4) / poly_5(x2, x4, x8, q0, q1, q2, q3, q4, q5);
            y1 = y1.mul_add(x2 * x, x);
        }

        if !bitmask.all() {
            y2 = ((one + x) / (one - x)).ln() * half;

            y2 = x
                .gt(one)
                .select(x.eq(one).select(Vf64::<S>::infinity(), Vf64::<S>::nan()), y2)
        }

        x_small.select(y1, y2).combine_sign(x0)
    }

    #[inline(always)]
    fn exp(x: Self::Vf) -> Self::Vf {
        exp_d_internal::<S>(x, ExpMode::Exp)
    }

    #[inline(always)]
    fn exph(x: Self::Vf) -> Self::Vf {
        exp_d_internal::<S>(x, ExpMode::Exph)
    }

    #[inline(always)]
    fn exp2(x: Self::Vf) -> Self::Vf {
        exp_d_internal::<S>(x, ExpMode::Pow2)
    }

    #[inline(always)]
    fn exp10(x: Self::Vf) -> Self::Vf {
        exp_d_internal::<S>(x, ExpMode::Pow10)
    }

    #[inline(always)]
    fn exp_m1(x: Self::Vf) -> Self::Vf {
        exp_d_internal::<S>(x, ExpMode::Expm1)
    }

    #[inline(always)]
    fn powf(x0: Self::Vf, y: Self::Vf) -> Self::Vf {
        // define constants
        let ln2d_hi = Vf64::<S>::splat(0.693145751953125); // log(2) in extra precision, high bits
        let ln2d_lo = Vf64::<S>::splat(1.42860682030941723212E-6); // low bits of log(2)
        let log2e = Vf64::<S>::splat(LOG2_E); // 1/log(2)
        let ln2 = Vf64::<S>::splat(LN_2);

        // coefficients for Pade polynomials
        let p0logl = Vf64::<S>::splat(2.0039553499201281259648E1);
        let p1logl = Vf64::<S>::splat(5.7112963590585538103336E1);
        let p2logl = Vf64::<S>::splat(6.0949667980987787057556E1);
        let p3logl = Vf64::<S>::splat(2.9911919328553073277375E1);
        let p4logl = Vf64::<S>::splat(6.5787325942061044846969E0);
        let p5logl = Vf64::<S>::splat(4.9854102823193375972212E-1);
        let p6logl = Vf64::<S>::splat(4.5270000862445199635215E-5);
        let q0logl = Vf64::<S>::splat(6.0118660497603843919306E1);
        let q1logl = Vf64::<S>::splat(2.1642788614495947685003E2);
        let q2logl = Vf64::<S>::splat(3.0909872225312059774938E2);
        let q3logl = Vf64::<S>::splat(2.2176239823732856465394E2);
        let q4logl = Vf64::<S>::splat(8.3047565967967209469434E1);
        let q5logl = Vf64::<S>::splat(1.5062909083469192043167E1);

        // Taylor coefficients for exp function, 1/n!
        let p2 = Vf64::<S>::splat(1.0 / 2.0);
        let p3 = Vf64::<S>::splat(1.0 / 6.0);
        let p4 = Vf64::<S>::splat(1.0 / 24.0);
        let p5 = Vf64::<S>::splat(1.0 / 120.0);
        let p6 = Vf64::<S>::splat(1.0 / 720.0);
        let p7 = Vf64::<S>::splat(1.0 / 5040.0);
        let p8 = Vf64::<S>::splat(1.0 / 40320.0);
        let p9 = Vf64::<S>::splat(1.0 / 362880.0);
        let p10 = Vf64::<S>::splat(1.0 / 3628800.0);
        let p11 = Vf64::<S>::splat(1.0 / 39916800.0);
        let p12 = Vf64::<S>::splat(1.0 / 479001600.0);
        let p13 = Vf64::<S>::splat(1.0 / 6227020800.0);

        let zero = Vf64::<S>::zero();
        let one = Vf64::<S>::one();
        let half = Vf64::<S>::splat(0.5);

        let x1 = x0.abs();

        let mut x = fraction2::<S>(x1);

        let blend = x.gt(Vf64::<S>::splat(SQRT_2 * 0.5));

        x += !blend.value() & x; // conditional add
        x -= one;

        let x2 = x * x;
        let x4 = x2 * x2;
        let x8 = x4 * x4;

        let px = x2 * x * poly_6(x, x2, x4, p0logl, p1logl, p2logl, p3logl, p4logl, p5logl, p6logl);
        let qx = poly_6(x, x2, x4, q0logl, q1logl, q2logl, q3logl, q4logl, q5logl, one);
        let lg1 = px / qx;

        let ef = exponent_f::<S>(x1) + (blend.value() & one);

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
        let mut v = e2.nmul_add(ln2d_lo, lg.mul_sub(y, e2 * ln2d_hi));

        // add remainder from ef * y
        v = yr.mul_add(ln2, v); // v += yr * VM_LN2;

        // correct for previous rounding errors
        v = (lgerr + x2err).nmul_add(y, v); // v -= (lgerr + x2err) * y;

        // extract something for the exponent if possible
        let mut x = v;
        let e3 = (x * log2e).round();

        // high precision multiplication not needed here because abs(e3) <= 1
        x = e3.nmul_add(ln2, x); // x -= e3 * VM_LN2;

        let x2 = x * x;
        let x4 = x2 * x2;
        let x8 = x4 * x4;

        // poly_13m + 1
        let mut z = poly_13(
            x, x2, x4, x8, one, one, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,
        );

        // contributions to exponent
        let ee = e1 + e2 + e3;
        let ei = unsafe { ee.to_int_fast() };

        // biased exponent of result:
        let ej = ei + Vi64::<S>::from_bits(z.into_bits()) >> 52;

        // check exponent for overflow and underflow
        let overflow = Vf64::<S>::from_cast_mask(ej.ge(Vi64::<S>::splat(0x07FF))) | ee.gt(Vf64::<S>::splat(3000.0));
        let underflow = Vf64::<S>::from_cast_mask(ej.le(Vi64::<S>::splat(0x0000))) | ee.lt(Vf64::<S>::splat(-3000.0));

        // add exponent by integer addition
        let mut z = Vf64::<S>::from_bits((ei.into_bits() << 52) + z.into_bits());

        // check for special cases
        let xfinite = x0.is_finite();
        let yfinite = y.is_finite();
        let efinite = ee.is_finite();

        let xzero = x0.is_zero_or_subnormal();
        let xsign = x0.is_negative();

        if unlikely!((overflow | underflow).any()) {
            z = underflow.select(zero, z);
            z = overflow.select(Vf64::<S>::infinity(), z);
        }

        let yzero = y.eq(zero);
        let yneg = y.lt(zero);

        // pow_case_x0
        z = xzero.select(yneg.select(Vf64::<S>::infinity(), yzero.select(one, zero)), z);

        let mut yodd = zero;

        if xsign.any() {
            let yint = y.eq(y.round());
            yodd = y << 63;

            let z1 = yint.select(z | yodd, x0.eq(zero).select(z, Vf64::<S>::nan()));

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
                .select(one, (x1.gt(one) ^ y.is_negative()).select(Vf64::<S>::infinity(), zero)),
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
        ln_d_internal::<S>(x, false)
    }

    #[inline(always)]
    fn ln_1p(x: Self::Vf) -> Self::Vf {
        ln_d_internal::<S>(x, true)
    }

    #[inline(always)]
    fn log2(x: Self::Vf) -> Self::Vf {
        x.ln() * Vf64::<S>::splat(LOG2_E)
    }

    #[inline(always)]
    fn log10(x: Self::Vf) -> Self::Vf {
        x.ln() * Vf64::<S>::splat(LOG10_E)
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn erf(x: Self::Vf) -> Self::Vf {
        // https://www.desmos.com/calculator/06q98crjp0
        let a0 = Vf64::<S>::one();
        let a1 = Vf64::<S>::splat(0.141047395888);
        let a2 = Vf64::<S>::splat(0.0895246554342);
        let a3 = Vf64::<S>::splat(0.024538446357);
        let a4 = Vf64::<S>::splat(0.00339526031482);
        let a5 = Vf64::<S>::splat(0.00127101693092);
        let a6 = Vf64::<S>::splat(0.000343596421733);
        let a7 = Vf64::<S>::splat(-0.0000282694821623);
        let a8 = Vf64::<S>::splat(0.0000153312079619);
        let a9 = Vf64::<S>::splat(0.00000806034527525);
        let a10 = Vf64::<S>::splat(-0.00000491119825703);
        let a11 = Vf64::<S>::splat(0.00000190850200269);
        let a12 = Vf64::<S>::splat(-4.5433487004e-7);
        let a13 = Vf64::<S>::splat(7.5111413853e-8);
        let a14 = Vf64::<S>::splat(-7.4944859806e-9);
        let a15 = Vf64::<S>::splat(3.8381832932e-10);

        let b = a0 - (a0 - x.abs()); // crush denormals
        let b2 = b * b;
        let b4 = b2 * b2;

        let r = poly_15(
            b, b2, b4, b4 * b4,
            a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
        );

        let r2 = r * r;
        let r4 = r2 * r2;
        let r8 = r4 * r4;

        (a0 - a0 / r8).copysign(x)
    }

    #[inline(always)]
    fn erfinv(y: Self::Vf) -> Self::Vf {
        let one = Vf64::<S>::one();

        let a = y.abs();

        let w = -a.nmul_add(a, one).ln();

        let mut p0 = {
            // https://www.desmos.com/calculator/06q98crjp0
            // TODO: Increase to 13?
            let c0 = Vf64::<S>::splat(1.50140935129);
            let c1 = Vf64::<S>::splat(0.246640278996);
            let c2 = Vf64::<S>::splat(-0.00417730548583);
            let c3 = Vf64::<S>::splat(-0.00125266932566);
            let c4 = Vf64::<S>::splat(0.00021832997994);
            let c5 = Vf64::<S>::splat(-0.00000488118158479);
            let c6 = Vf64::<S>::splat(-0.0000032971414425);
            let c7 = Vf64::<S>::splat(5.6978210432e-7);
            let c8 = Vf64::<S>::splat(6.435368102e-8);
            let c9 = Vf64::<S>::splat(9.7659875934e-9);
            let c10 = Vf64::<S>::splat(6.7370001038e-9);
            let c11 = Vf64::<S>::splat(1.1765533188e-9);
            let c12 = Vf64::<S>::splat(6.4996699923e-11);

            let w1 = w - Vf64::<S>::splat(2.5);
            let w2 = w1 * w1;
            let w4 = w2 * w2;
            let w8 = w4 * w4;

            poly_12(w1, w2, w4, w8, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12)
        };

        let w_big = w.ge(Vf64::<S>::splat(5.0)); // at around |x| > 0.99662533231, so unlikely

        if unlikely!(w_big.any()) {
            let c0 = Vf64::<S>::splat(3.05596299225);
            let c1 = Vf64::<S>::splat(2.48600156101);
            let c2 = Vf64::<S>::splat(4.49503793389);
            let c3 = Vf64::<S>::splat(8.08173608956);
            let c4 = Vf64::<S>::splat(9.59962183284);
            let c5 = Vf64::<S>::splat(7.71451614152);
            let c6 = Vf64::<S>::splat(4.10730178764);
            let c7 = Vf64::<S>::splat(1.22408705176);
            let c8 = Vf64::<S>::splat(-0.0243566098554);
            let c9 = Vf64::<S>::splat(-0.217358699758);
            let c10 = Vf64::<S>::splat(-0.113869576373);
            let c11 = Vf64::<S>::splat(-0.0338146066555);
            let c12 = Vf64::<S>::splat(-0.00648455409479);
            let c13 = Vf64::<S>::splat(-0.000799937382768);
            let c14 = Vf64::<S>::splat(-0.0000582036485279);
            let c15 = Vf64::<S>::splat(-0.00000190953540332);

            let w1 = w.sqrt() - Vf64::<S>::splat(3.0);
            let w2 = w1 * w1;
            let w4 = w2 * w2;
            let w8 = w4 * w4;

            let mut p1 = poly_15(
                w1, w2, w4, w8, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15,
            );

            p1 = a.eq(one).select(Vf64::<S>::infinity(), p1); // erfinv(x == 1) = inf
            p1 = a.gt(one).select(Vf64::<S>::nan(), p1); // erfinv(x > 1) = NaN

            p0 = w_big.select(p1, p0);
        }

        p0 * y
    }

    #[inline(always)]
    fn next_float(x: Self::Vf) -> Self::Vf {
        let i1 = Vu64::<S>::one();

        let v = x.eq(Vf64::<S>::neg_zero()).select(Vf64::<S>::zero(), x);

        let bits = v.into_bits();
        x.eq(Vf64::<S>::infinity()).select(
            x,
            Vf64::<S>::from_bits(v.ge(Vf64::<S>::zero()).select(bits + i1, bits - i1)),
        )
    }

    #[inline(always)]
    fn prev_float(x: Self::Vf) -> Self::Vf {
        let i1 = Vu64::<S>::one();

        let v = x.eq(Vf64::<S>::zero()).select(Vf64::<S>::neg_zero(), x);

        let bits = v.into_bits();
        x.eq(Vf64::<S>::neg_infinity()).select(
            x,
            Vf64::<S>::from_bits(v.gt(Vf64::<S>::zero()).select(bits - i1, bits + i1)),
        )
    }
}

#[inline(always)]
fn fraction2<S: Simd>(x: Vf64<S>) -> Vf64<S> {
    // set exponent to 0 + bias
    (x & Vf64::<S>::splat(f64::from_bits(0x000FFFFFFFFFFFFF))) | Vf64::<S>::splat(f64::from_bits(0x3FE0000000000000))
}

#[inline(always)]
fn exponent<S: Simd>(x: Vf64<S>) -> Vi32<S> {
    // shift out sign, extract exp, subtract bias
    Vi32::<S>::from_bits(<Vu32<S> as SimdCastFrom<S, Vu64<S>>>::from_cast(
        (x.into_bits() << 1) >> 53,
    )) - Vi32::<S>::splat(0x3FF)
}

#[inline(always)]
fn exponent_f<S: Simd>(x: Vf64<S>) -> Vf64<S> {
    let pow2_52 = Vf64::<S>::splat(4503599627370496.0);
    let bias = Vf64::<S>::splat(1023.0);

    Vf64::<S>::from_bits((x.into_bits() >> 52) | pow2_52.into_bits()) - (pow2_52 + bias)
}

#[inline(always)]
fn ln_d_internal<S: Simd>(x0: Vf64<S>, p1: bool) -> Vf64<S> {
    let ln2_hi = Vf64::<S>::splat(0.693359375);
    let ln2_lo = Vf64::<S>::splat(-2.121944400546905827679E-4);
    let p0log = Vf64::<S>::splat(7.70838733755885391666E0);
    let p1log = Vf64::<S>::splat(1.79368678507819816313E1);
    let p2log = Vf64::<S>::splat(1.44989225341610930846E1);
    let p3log = Vf64::<S>::splat(4.70579119878881725854E0);
    let p4log = Vf64::<S>::splat(4.97494994976747001425E-1);
    let p5log = Vf64::<S>::splat(1.01875663804580931796E-4);
    let q0log = Vf64::<S>::splat(2.31251620126765340583E1);
    let q1log = Vf64::<S>::splat(7.11544750618563894466E1);
    let q2log = Vf64::<S>::splat(8.29875266912776603211E1);
    let q3log = Vf64::<S>::splat(4.52279145837532221105E1);
    let q4log = Vf64::<S>::splat(1.12873587189167450590E1);

    let one = Vf64::<S>::one();
    let zero = Vf64::<S>::zero();

    let x1 = if p1 { x0 + one } else { x0 };

    let mut x = fraction2::<S>(x1);
    let mut fe = <Vf64<S> as SimdCastFrom<S, Vi32<S>>>::from_cast(exponent::<S>(x1));

    let blend = x.gt(Vf64::<S>::splat(SQRT_2 * 0.5));

    // conditional adds
    x += !blend.value() & x;
    fe += blend.value() & one;

    let xp1 = x - one;

    x = if p1 {
        // log(x+1). Avoid loss of precision when adding 1 and later subtracting 1 if exponent = 0
        fe.eq(zero).select(x0, xp1)
    } else {
        // log(x). Expand around 1.0
        xp1
    };

    let x2 = x * x;
    let x3 = x * x2;
    let x4 = x2 * x2;

    let px = poly_5(x, x2, x4, p0log, p1log, p2log, p3log, p4log, p5log) * x3;
    let qx = poly_5(x, x2, x4, q0log, q1log, q2log, q3log, q4log, one);

    let mut res = px / qx;

    res = fe.mul_add(ln2_lo, res); // res += fe * ln2_lo;
    res += x2.nmul_add(Vf64::<S>::splat(0.5), x); // res += x - 0.5 * x2;
    res = fe.mul_add(ln2_hi, res); // res += fe * ln2_hi;

    let overflow = !x1.is_finite();
    let underflow = x1.lt(Vf64::<S>::splat(2.2250738585072014E-308));

    if likely!((overflow | underflow).none()) {
        return res;
    }

    res = underflow.select(Vf64::<S>::nan(), res); // x1 < 0 gives NAN
    res = x1.is_zero_or_subnormal().select(Vf64::<S>::neg_infinity(), res); // x1 == 0 gives -INF
    res = overflow.select(x1, res); // INF or NAN goes through
    res = (x1.is_infinite() & x1.is_negative()).select(Vf64::<S>::nan(), res); // -INF gives NAN

    res
}

#[inline(always)]
fn atan_internal<S: Simd>(y: Vf64<S>, x: Vf64<S>, atan2: bool) -> Vf64<S> {
    let morebits = Vf64::<S>::splat(6.123233995736765886130E-17);
    let morebitso2 = Vf64::<S>::splat(6.123233995736765886130E-17 * 0.5);
    let t3po8 = Vf64::<S>::splat(SQRT_2 + 1.0);
    let p4atan = Vf64::<S>::splat(-8.750608600031904122785E-1);
    let p3atan = Vf64::<S>::splat(-1.615753718733365076637E1);
    let p2atan = Vf64::<S>::splat(-7.500855792314704667340E1);
    let p1atan = Vf64::<S>::splat(-1.228866684490136173410E2);
    let p0atan = Vf64::<S>::splat(-6.485021904942025371773E1);
    let q4atan = Vf64::<S>::splat(2.485846490142306297962E1);
    let q3atan = Vf64::<S>::splat(1.650270098316988542046E2);
    let q2atan = Vf64::<S>::splat(4.328810604912902668951E2);
    let q1atan = Vf64::<S>::splat(4.853903996359136964868E2);
    let q0atan = Vf64::<S>::splat(1.945506571482613964425E2);
    let neg_one = Vf64::<S>::neg_one();
    let one = Vf64::<S>::one();
    let zero = Vf64::<S>::zero();

    let mut swapxy = Mask::new(unsafe { Vf64::<S>::undefined() });

    let t = if atan2 {
        let x1 = x.abs();
        let y1 = y.abs();

        swapxy = y1.gt(x1);

        let mut x2 = swapxy.select(y1, x1);
        let mut y2 = swapxy.select(x1, y1);

        let both_inf = x.is_infinite() & y.is_infinite();

        // TODO: Benchmark this branch
        if unlikely!(both_inf.any()) {
            x2 = both_inf.select(x2 & neg_one, x2);
            y2 = both_inf.select(y2 & neg_one, y2);
        }

        y2 / x2
    } else {
        y.abs()
    };

    let not_big = t.le(t3po8);
    let not_small = t.ge(Vf64::<S>::splat(0.66));

    let s = not_big.select(Vf64::<S>::splat(FRAC_PI_4), Vf64::<S>::splat(FRAC_PI_2)) & not_small.value();

    let fac = not_big.select(morebitso2, morebits) & not_small.value();

    let a = (not_big.value() & t) + (not_small.value() & neg_one);
    let b = (not_big.value() & one) + (not_small.value() & t);

    let z = a / b;

    let zz = z * z;
    let zz2 = zz * zz;
    let zz4 = zz2 * zz2;

    let px = poly_4(zz, zz2, zz4, p0atan, p1atan, p2atan, p3atan, p4atan);
    let qx = poly_5(zz, zz2, zz4, q0atan, q1atan, q2atan, q3atan, q4atan, one);

    // place additions before mul_add to lessen dependency chain
    let mut re = (px / qx).mul_add(z * zz, z + s + fac);

    if atan2 {
        re = swapxy.select(Vf64::<S>::splat(FRAC_PI_2) - re, re);
        re = (x | y).eq(zero).select(zero, re); // atan2(0,0) = 0 by convention
                                                // also for x = -0.
        re = x.is_negative().select(Vf64::<S>::splat(PI) - re, re);
    }

    re.combine_sign(y)
}

#[inline(always)]
fn asin_internal<S: Simd>(x: Vf64<S>, acos: bool) -> Vf64<S> {
    let r4asin = Vf64::<S>::splat(2.967721961301243206100E-3);
    let r3asin = Vf64::<S>::splat(-5.634242780008963776856E-1);
    let r2asin = Vf64::<S>::splat(6.968710824104713396794E0);
    let r1asin = Vf64::<S>::splat(-2.556901049652824852289E1);
    let r0asin = Vf64::<S>::splat(2.853665548261061424989E1);
    let s3asin = Vf64::<S>::splat(-2.194779531642920639778E1);
    let s2asin = Vf64::<S>::splat(1.470656354026814941758E2);
    let s1asin = Vf64::<S>::splat(-3.838770957603691357202E2);
    let s0asin = Vf64::<S>::splat(3.424398657913078477438E2);
    let p5asin = Vf64::<S>::splat(4.253011369004428248960E-3);
    let p4asin = Vf64::<S>::splat(-6.019598008014123785661E-1);
    let p3asin = Vf64::<S>::splat(5.444622390564711410273E0);
    let p2asin = Vf64::<S>::splat(-1.626247967210700244449E1);
    let p1asin = Vf64::<S>::splat(1.956261983317594739197E1);
    let p0asin = Vf64::<S>::splat(-8.198089802484824371615E0);
    let q4asin = Vf64::<S>::splat(-1.474091372988853791896E1);
    let q3asin = Vf64::<S>::splat(7.049610280856842141659E1);
    let q2asin = Vf64::<S>::splat(-1.471791292232726029859E2);
    let q1asin = Vf64::<S>::splat(1.395105614657485689735E2);
    let q0asin = Vf64::<S>::splat(-4.918853881490881290097E1);
    let one = Vf64::<S>::one();

    let xa = x.abs();

    let is_big = xa.ge(Vf64::<S>::splat(0.625));

    let x1 = is_big.select(one - xa, xa * xa);

    let x2 = x1 * x1;
    let x4 = x2 * x2;
    let x8 = x4 * x4;

    let undef = unsafe { Vf64::<S>::undefined() };

    let mut px = undef;
    let mut qx = undef;
    let mut rx = undef;
    let mut sx = undef;
    let mut xb = undef;

    let bitmask = is_big.bitmask();

    // if not all are big (if any are small)
    if !bitmask.all() {
        px = poly_5(x1, x2, x4, p0asin, p1asin, p2asin, p3asin, p4asin, p5asin);
        qx = poly_5(x1, x2, x4, q0asin, q1asin, q2asin, q3asin, q4asin, one);
    }

    // if any are big
    if bitmask.any() {
        rx = poly_4(x1, x2, x4, r0asin, r1asin, r2asin, r3asin, r4asin);
        sx = poly_4(x1, x2, x4, s0asin, s1asin, s2asin, s3asin, one);
        xb = (x1 + x1).sqrt();
    }

    let vx = is_big.select(rx, px);
    let wx = is_big.select(sx, qx);

    let y1 = vx / wx * x1;

    // avoid branching again for this single instruction, just do it
    let z1 = xb.mul_add(y1, xb);
    let z2 = xa.mul_add(y1, xa);

    let frac_pi_2 = Vf64::<S>::splat(FRAC_PI_2);

    if acos {
        let z1 = x.is_negative().select(Vf64::<S>::splat(PI) - z1, z1);
        let z2 = frac_pi_2 - z2.combine_sign(x);
        is_big.select(z1, z2)
    } else {
        let z1 = frac_pi_2 - z1;
        is_big.select(z1, z2).combine_sign(x)
    }
}

#[inline(always)]
fn pow2n_d<S: Simd>(n: Vf64<S>) -> Vf64<S> {
    let pow2_52 = Vf64::<S>::splat(4503599627370496.0);
    let bias = Vf64::<S>::splat(1023.0);

    (n + (bias + pow2_52)) << 52
}

#[inline(always)]
fn exp_d_internal<S: Simd>(x0: Vf64<S>, mode: ExpMode) -> Vf64<S> {
    let zero = Vf64::<S>::zero();
    let one = Vf64::<S>::one();

    // Taylor coefficients, 1/n!
    // Not using minimax approximation because we prioritize precision close to x = 0
    let p0 = zero;
    let p1 = one;
    let p2 = Vf64::<S>::splat(1.0 / 2.0);
    let p3 = Vf64::<S>::splat(1.0 / 6.0);
    let p4 = Vf64::<S>::splat(1.0 / 24.0);
    let p5 = Vf64::<S>::splat(1.0 / 120.0);
    let p6 = Vf64::<S>::splat(1.0 / 720.0);
    let p7 = Vf64::<S>::splat(1.0 / 5040.0);
    let p8 = Vf64::<S>::splat(1.0 / 40320.0);
    let p9 = Vf64::<S>::splat(1.0 / 362880.0);
    let p10 = Vf64::<S>::splat(1.0 / 3628800.0);
    let p11 = Vf64::<S>::splat(1.0 / 39916800.0);
    let p12 = Vf64::<S>::splat(1.0 / 479001600.0);
    let p13 = Vf64::<S>::splat(1.0 / 6227020800.0);

    let mut x = x0;
    let mut r;

    let max_x;

    match mode {
        ExpMode::Exp | ExpMode::Exph | ExpMode::Expm1 => {
            max_x = if mode == ExpMode::Exp { 708.39 } else { 709.7 };

            let ln2d_hi = Vf64::<S>::splat(0.693145751953125);
            let ln2d_lo = Vf64::<S>::splat(1.42860682030941723212E-6);

            r = (x0 * Vf64::<S>::splat(LOG2_E)).round();

            x = r.nmul_add(ln2d_hi, x); // x -= r * ln2_hi;
            x = r.nmul_add(ln2d_lo, x); // x -= r * ln2_lo;

            if mode == ExpMode::Exph {
                r -= one;
            }
        }
        ExpMode::Pow2 => {
            max_x = 1022.0;

            r = x0.round();

            x -= r;
            x *= Vf64::<S>::splat(LN_2);
        }
        ExpMode::Pow10 => {
            max_x = 307.65;

            let log10_2_hi = Vf64::<S>::splat(0.30102999554947019); // log10(2) in two parts
            let log10_2_lo = Vf64::<S>::splat(1.1451100899212592E-10);

            r = (x0 * Vf64::<S>::splat(LN_10 * LOG2_E)).round();

            x = r.nmul_add(log10_2_hi, x); // x -= r * log10_2_hi;
            x = r.nmul_add(log10_2_lo, x); // x -= r * log10_2_lo;
            x *= Vf64::<S>::splat(LN_10);
        }
    }

    let x2 = x * x;
    let x4 = x2 * x2;
    let x8 = x4 * x4;

    let mut z = poly_13(
        x, x2, x4, x8, p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13,
    );

    let n2 = pow2n_d::<S>(r);

    if mode == ExpMode::Expm1 {
        z = z.mul_add(n2, n2 - one);
    } else {
        z = z.mul_add(n2, n2); // (z + 1.0f) * n2
    }

    let in_range = x0.abs().lt(Vf64::<S>::splat(max_x)) & x0.is_finite().cast_to();

    if likely!(in_range.all()) {
        return z;
    }

    let sign_bit_mask = (x0 & Vf64::<S>::neg_zero()).into_bits().ne(Vu64::<S>::zero());
    let is_nan = x0.is_nan();

    let underflow_value = if mode == ExpMode::Expm1 {
        Vf64::<S>::neg_one()
    } else {
        Vf64::<S>::zero()
    };

    r = sign_bit_mask.select(underflow_value, Vf64::<S>::infinity());
    z = in_range.select(z, r);
    z = is_nan.select(x0, z);

    z
}
