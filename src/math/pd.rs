use super::{poly::*, *};

use core::f64::consts::{FRAC_PI_2, FRAC_PI_4, LN_10, LN_2, LOG10_2, LOG10_E, LOG2_E, PI, SQRT_2};

impl<S: Simd> SimdVectorizedMathInternal<S> for f64
where
    <S as Simd>::Vf64: SimdFloatVector<S, Element = f64>,
{
    type Vf = <S as Simd>::Vf64;

    const __EPSILON: Self = f64::EPSILON;

    #[inline(always)]
    fn sin_cos<P: Policy>(xx: Self::Vf) -> (Self::Vf, Self::Vf) {
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

        if P::POLICY.check_overflow() {
            let overflow = y.gt(Vf64::<S>::splat((1u64 << 52) as f64 - 1.0)) & xa.is_finite();

            let s = overflow.select(zero, s);
            let c = overflow.select(one, c);
        }

        let sin1 = swap.select(c, s);
        let cos1 = swap.select(s, c);

        let signsin = Vf64::<S>::from_bits((q << 62)) ^ xx;
        let signcos = Vf64::<S>::from_bits(((q + Vu64::<S>::one()) & Vu64::<S>::splat(2)) << 62);

        // combine signs
        (sin1.combine_sign(signsin), cos1 ^ signcos)
    }

    #[inline(always)]
    fn asin<P: Policy>(x: Self::Vf) -> Self::Vf {
        asin_internal::<S, P>(x, false)
    }

    #[inline(always)]
    fn acos<P: Policy>(x: Self::Vf) -> Self::Vf {
        asin_internal::<S, P>(x, true)
    }

    #[inline(always)]
    fn atan<P: Policy>(x: Self::Vf) -> Self::Vf {
        atan_internal::<S, P>(x, unsafe { Vf64::<S>::undefined() }, false)
    }
    #[inline(always)]
    fn atan2<P: Policy>(y: Self::Vf, x: Self::Vf) -> Self::Vf {
        atan_internal::<S, P>(y, x, true)
    }

    #[inline(always)]
    fn sinh<P: Policy>(x0: Self::Vf) -> Self::Vf {
        let one = Vf64::<S>::one();

        let x = x0.abs();

        let x_small = x.le(one);

        let mut y1 = unsafe { Vf64::<S>::undefined() };
        let mut y2 = unsafe { Vf64::<S>::undefined() };

        // use bitmask directly to avoid two calls
        let bitmask = x_small.bitmask();

        // if any are small
        if bitmask.any() {
            let p0 = Vf64::<S>::splat(-3.51754964808151394800E5);
            let p1 = Vf64::<S>::splat(-1.15614435765005216044E4);
            let p2 = Vf64::<S>::splat(-1.63725857525983828727E2);
            let p3 = Vf64::<S>::splat(-7.89474443963537015605E-1);
            let q0 = Vf64::<S>::splat(-2.11052978884890840399E6);
            let q1 = Vf64::<S>::splat(3.61578279834431989373E4);
            let q2 = Vf64::<S>::splat(-2.77711081420602794433E2);
            let q3 = one;

            let x2 = x * x;
            let x4 = x2 * x2;

            y1 = poly_3(x2, x4, p0, p1, p2, p3) / poly_3(x2, x4, q0, q1, q2, q3);
            y1 = y1.mul_add(x * x2, x);
        }

        // if not all are small
        if !bitmask.all() {
            y2 = Self::exph::<P>(x);
            y2 -= Vf64::<S>::splat(0.25) / y2;
        }

        x_small.select(y1, y2).combine_sign(x0)
    }

    #[inline(always)]
    fn tanh<P: Policy>(x0: Self::Vf) -> Self::Vf {
        let one = Vf64::<S>::one();

        let x = x0.abs();

        let x_small = x.le(Vf64::<S>::splat(0.625));

        let mut y1 = unsafe { Vf64::<S>::undefined() };
        let mut y2 = unsafe { Vf64::<S>::undefined() };

        // use bitmask directly to avoid two calls
        let bitmask = x_small.bitmask();

        // if any are small
        if bitmask.any() {
            let p0 = Vf64::<S>::splat(-1.61468768441708447952E3);
            let p1 = Vf64::<S>::splat(-9.92877231001918586564E1);
            let p2 = Vf64::<S>::splat(-9.64399179425052238628E-1);
            let q0 = Vf64::<S>::splat(4.84406305325125486048E3);
            let q1 = Vf64::<S>::splat(2.23548839060100448583E3);
            let q2 = Vf64::<S>::splat(1.12811678491632931402E2);
            let q3 = one;

            let x2 = x * x;
            let x4 = x2 * x2;

            y1 = poly_2(x2, x4, p0, p1, p2) / poly_3(x2, x4, q0, q1, q2, q3);
            y1 = y1.mul_add(x2 * x, x);
        }

        // if not all are small
        if !bitmask.all() {
            y2 = Self::exp::<P>(x + x);
            y2 = (y2 - one) / (y2 + one); // originally (1 - 2/(y2 + 1)), but doing it this way avoids loading 2.0
        }

        let x_big = x.gt(Vf64::<S>::splat(350.0));

        y1 = x_small.select(y1, y2);
        y1 = x_big.select(one, y1);

        y1.combine_sign(x0)
    }

    #[inline(always)]
    fn asinh<P: Policy>(x0: Self::Vf) -> Self::Vf {
        let one = Vf64::<S>::one();

        let x = x0.abs();
        let x2 = x0 * x0;

        let x_small = x.le(Vf64::<S>::splat(0.533));
        let x_huge = x.gt(Vf64::<S>::splat(1e20));

        let mut y1 = unsafe { Vf64::<S>::undefined() };
        let mut y2 = unsafe { Vf64::<S>::undefined() };

        let bitmask = x_small.bitmask();

        if bitmask.any() {
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

            let x4 = x2 * x2;
            let x8 = x4 * x4;

            y1 = poly_4(x2, x4, x8, p0, p1, p2, p3, p4) / poly_4(x2, x4, x8, q0, q1, q2, q3, q4);
            y1 = y1.mul_add(x2 * x, x);
        }

        if !bitmask.all() {
            y2 = Self::ln::<P>((x2 + one).sqrt() + x);

            if unlikely!(x_huge.any()) {
                y2 = x_huge.select(Self::ln::<P>(x) + Vf64::<S>::splat(LN_2), y2);
            }
        }

        x_small.select(y1, y2).combine_sign(x0)
    }

    #[inline(always)]
    fn acosh<P: Policy>(x0: Self::Vf) -> Self::Vf {
        let one = Vf64::<S>::one();

        let x1 = x0 - one;

        let is_undef = x0.lt(one);
        let x_small = x1.lt(Vf64::<S>::splat(0.49));
        let x_huge = x1.gt(Vf64::<S>::splat(1e20));

        let mut y1 = unsafe { Vf64::<S>::undefined() };
        let mut y2 = unsafe { Vf64::<S>::undefined() };

        let bitmask = x_small.bitmask();

        if bitmask.any() {
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

            let x2 = x1 * x1;
            let x4 = x2 * x2;

            y1 = x1.sqrt() * (poly_4(x1, x2, x4, p0, p1, p2, p3, p4) / poly_5(x1, x2, x4, q0, q1, q2, q3, q4, q5));
            y1 = is_undef.select(Vf64::<S>::nan(), y1);
        }

        if !bitmask.all() {
            y2 = Self::ln::<P>(x0.mul_sub(x0, one).sqrt() + x0);

            if unlikely!(x_huge.any()) {
                y2 = x_huge.select(Self::ln::<P>(x0) + Vf64::<S>::splat(LN_2), y2);
            }
        }

        x_small.select(y1, y2)
    }

    #[inline(always)]
    fn atanh<P: Policy>(x0: Self::Vf) -> Self::Vf {
        let one = Vf64::<S>::one();
        let half = Vf64::<S>::splat(0.5);

        let x = x0.abs();

        let x_small = x.lt(half);

        let mut y1 = unsafe { Vf64::<S>::undefined() };
        let mut y2 = unsafe { Vf64::<S>::undefined() };

        let bitmask = x_small.bitmask();

        if bitmask.any() {
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

            let x2 = x * x;
            let x4 = x2 * x2;
            let x8 = x4 * x4;

            y1 = poly_4(x2, x4, x8, p0, p1, p2, p3, p4) / poly_5(x2, x4, x8, q0, q1, q2, q3, q4, q5);
            y1 = y1.mul_add(x2 * x, x);
        }

        if !bitmask.all() {
            y2 = Self::ln::<P>((one + x) / (one - x)) * half;

            y2 = x
                .gt(one)
                .select(x.eq(one).select(Vf64::<S>::infinity(), Vf64::<S>::nan()), y2)
        }

        x_small.select(y1, y2).combine_sign(x0)
    }

    #[inline(always)]
    fn exp<P: Policy>(x: Self::Vf) -> Self::Vf {
        exp_d_internal::<S, P>(x, ExpMode::Exp)
    }

    #[inline(always)]
    fn exph<P: Policy>(x: Self::Vf) -> Self::Vf {
        exp_d_internal::<S, P>(x, ExpMode::Exph)
    }

    #[inline(always)]
    fn exp2<P: Policy>(x: Self::Vf) -> Self::Vf {
        exp_d_internal::<S, P>(x, ExpMode::Pow2)
    }

    #[inline(always)]
    fn exp10<P: Policy>(x: Self::Vf) -> Self::Vf {
        exp_d_internal::<S, P>(x, ExpMode::Pow10)
    }

    #[inline(always)]
    fn exp_m1<P: Policy>(x: Self::Vf) -> Self::Vf {
        exp_d_internal::<S, P>(x, ExpMode::Expm1)
    }

    #[inline(always)]
    fn cbrt<P: Policy>(x: Self::Vf) -> Self::Vf {
        let b1 = Vu64::<S>::splat(715094163); // B1 = (1023-1023/3-0.03306235651)*2**20
        let b2 = Vu64::<S>::splat(696219795); // B2 = (1023-1023/3-54/3-0.03306235651)*2**20
        let m = Vu64::<S>::splat(0x7fffffff); // u32::MAX >> 1

        let x1p54 = x * Vf64::<S>::splat(f64::from_bits(0x4350000000000000)); // 0x1p54 === 2 ^ 54

        let hx0 = (x.into_bits() >> 32) & m;

        let x_small = hx0.lt(Vu64::<S>::splat(0x00100000));

        let xs = x_small.select(x1p54, x); // note that this upcasts
        let b = x_small.select(b2, b1);

        let mut ui = xs.into_bits();
        let mut hx = (ui >> 32) & m;

        // TODO: Fix this when stable isn't broken
        hx = <Vu64<S> as Div<Divider<u64>>>::div(hx, Divider::u64(3)) + b;

        ui &= Vu64::<S>::splat(1 << 63);
        ui |= hx << 32;

        let mut t = Vf64::<S>::from_bits(ui);

        let r = (t * t) * (t / x);
        let r2 = r * r;

        t *= r.poly(&[
            1.87595182427177009643,   /* 0x3ffe03e6, 0x0f61e692 */
            -1.88497979543377169875,  /* 0xbffe28e0, 0x92f02420 */
            1.621429720105354466140,  /* 0x3ff9f160, 0x4a49d6c2 */
            -0.758397934778766047437, /* 0xbfe844cb, 0xbee751d9 */
            0.145996192886612446982,  /* 0x3fc2b000, 0xd4e4edd7 */
        ]);

        ui = t.into_bits();
        ui = (ui + Vu64::<S>::splat(0x80000000)) & Vu64::<S>::splat(0xffffffffc0000000);
        t = Vf64::<S>::from_bits(ui);

        // TODO: Policy for non-FMA handling

        // original form, 5 simple ops, 2 divisions
        //let r = ((x / (t * t)) - t) / ((t + t) + (x / (t * t)));

        // fast form, 3 simple ops, 1 division, 1 fma
        let t3 = t * t * t;
        let r = (x - t3) / Vf64::<S>::splat(2.0).mul_add(t3, x);

        t = r.mul_add(t, t);

        (hx0.gt(Vu64::<S>::splat(0x7f800000)) | hx0.eq(Vu64::<S>::zero())).select(x, t)
    }

    #[inline(always)]
    fn powf<P: Policy>(x0: Self::Vf, y: Self::Vf) -> Self::Vf {
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

        // poly_13m + 1, Taylor coefficients for exp function, 1/n!
        let mut z = x.poly(&[
            1.0, // + 1
            1.0 / 1.0,
            1.0 / 2.0,
            1.0 / 6.0,
            1.0 / 24.0,
            1.0 / 120.0,
            1.0 / 720.0,
            1.0 / 5040.0,
            1.0 / 40320.0,
            1.0 / 362880.0,
            1.0 / 3628800.0,
            1.0 / 39916800.0,
            1.0 / 479001600.0,
            1.0 / 6227020800.0,
        ]);

        // contributions to exponent
        let ee = e1 + e2 + e3;
        let ei = unsafe { ee.to_int_fast() };

        // biased exponent of result:
        let ej = ei + Vi64::<S>::from_bits(z.into_bits()) >> 52;

        // add exponent by integer addition
        let mut z = Vf64::<S>::from_bits((ei.into_bits() << 52) + z.into_bits());

        if !P::POLICY.check_overflow() {
            return z;
        }

        // check exponent for overflow and underflow
        let overflow = Vf64::<S>::from_cast_mask(ej.ge(Vi64::<S>::splat(0x07FF))) | ee.gt(Vf64::<S>::splat(3000.0));
        let underflow = Vf64::<S>::from_cast_mask(ej.le(Vi64::<S>::splat(0x0000))) | ee.lt(Vf64::<S>::splat(-3000.0));

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
    fn ln<P: Policy>(x: Self::Vf) -> Self::Vf {
        ln_d_internal::<S, P>(x, false)
    }

    #[inline(always)]
    fn ln_1p<P: Policy>(x: Self::Vf) -> Self::Vf {
        ln_d_internal::<S, P>(x, true)
    }

    #[inline(always)]
    fn log2<P: Policy>(x: Self::Vf) -> Self::Vf {
        ln_d_internal::<S, P>(x, false) * Vf64::<S>::splat(LOG2_E)
    }

    #[inline(always)]
    fn log10<P: Policy>(x: Self::Vf) -> Self::Vf {
        ln_d_internal::<S, P>(x, false) * Vf64::<S>::splat(LOG10_E)
    }

    #[rustfmt::skip]
    #[inline(always)]
    fn erf<P: Policy>(x: Self::Vf) -> Self::Vf {
        let p0 = Vf64::<S>::splat(5.55923013010394962768e4);
        let p1 = Vf64::<S>::splat(7.00332514112805075473e3);
        let p2 = Vf64::<S>::splat(2.23200534594684319226e3);
        let p3 = Vf64::<S>::splat(9.00260197203842689217e1);
        let p4 = Vf64::<S>::splat(9.60497373987051638749e0);

        let q0 = Vf64::<S>::splat(4.92673942608635921086e4);
        let q1 = Vf64::<S>::splat(2.26290000613890934246e4);
        let q2 = Vf64::<S>::splat(4.59432382970980127987e3);
        let q3 = Vf64::<S>::splat(5.21357949780152679795e2);
        let q4 = Vf64::<S>::splat(3.35617141647503099647e1);
        let q5 = Vf64::<S>::splat(1.00000000000000000000e0);

        let z1 = x * x;
        let z2 = z1 * z1;
        let z4 = z2 * z2;

        x * poly_4(z1, z2, z4, p0, p1, p2, p3, p4) /
            poly_5(z1, z2, z4, q0, q1, q2, q3, q4, q5)
    }

    #[inline(always)]
    fn erfinv<P: Policy>(y: Self::Vf) -> Self::Vf {
        let one = Vf64::<S>::one();

        let a = y.abs();

        let w = -Self::ln::<P>(a.nmul_add(a, one));

        // https://www.desmos.com/calculator/yduhxx1ukm values extracted via JS console
        let mut p0 = (w - Vf64::<S>::splat(2.5)).poly(&[
            1.501409350414994,
            0.2466402709383954,
            -0.0041773392840529855,
            -0.001252754693878528,
            0.00021818504236422313,
            -0.000005055953518603739,
            -0.000003451228003698613,
            4.691555466910589e-7,
            1.565009183876413e-8,
            -7.498144332533493e-9,
            2.378447620687541e-9,
            4.340759057762667e-10,
            -1.1526825105953649e-11,
            -3.605158594283844e-12,
        ]);

        let w_big = w.ge(Vf64::<S>::splat(5.0)); // at around |x| > 0.99662533231, so unlikely

        if unlikely!(w_big.any()) {
            let mut p1 = (w.sqrt() - Vf64::<S>::splat(3.0)).poly(&[
                2.914513093490991,
                1.5466942804733321,
                1.5950004257395263,
                2.559965578101086,
                2.3489887347568135,
                0.7600225853251197,
                -0.9258061028319879,
                -1.574375166164548,
                -1.2294848322739875,
                -0.6192716293714041,
                -0.21681459128064842,
                -0.05369968979686224,
                -0.009288117987439485,
                -0.0010722580888930223,
                -0.00007449590390143766,
                -0.0000023620166848468398,
            ]);

            if P::POLICY.check_overflow() {
                p1 = a.eq(one).select(Vf64::<S>::infinity(), p1); // erfinv(x == 1) = inf
                p1 = a.gt(one).select(Vf64::<S>::nan(), p1); // erfinv(x > 1) = NaN
            }

            p0 = w_big.select(p1, p0);
        }

        p0 * y
    }

    #[inline(always)]
    fn next_float<P: Policy>(x: Self::Vf) -> Self::Vf {
        let i1 = Vu64::<S>::one();

        let v = x.eq(Vf64::<S>::neg_zero()).select(Vf64::<S>::zero(), x);

        let bits = v.into_bits();
        x.eq(Vf64::<S>::infinity()).select(
            x,
            Vf64::<S>::from_bits(v.ge(Vf64::<S>::zero()).select(bits + i1, bits - i1)),
        )
    }

    #[inline(always)]
    fn prev_float<P: Policy>(x: Self::Vf) -> Self::Vf {
        let i1 = Vu64::<S>::one();

        let v = x.eq(Vf64::<S>::zero()).select(Vf64::<S>::neg_zero(), x);

        let bits = v.into_bits();
        x.eq(Vf64::<S>::neg_infinity()).select(
            x,
            Vf64::<S>::from_bits(v.gt(Vf64::<S>::zero()).select(bits - i1, bits + i1)),
        )
    }

    #[inline(always)]
    fn tgamma<P: Policy>(mut z: Self::Vf) -> Self::Vf {
        let zero = Vf64::<S>::zero();
        let one = Vf64::<S>::one();
        let half = Vf64::<S>::splat(0.5);
        let quarter = Vf64::<S>::splat(0.25);
        let pi = Vf64::<S>::splat(PI);

        let orig_z = z;

        let is_neg = z.is_negative();
        let mut reflected = Mask::falsey();

        let mut res = one;

        'goto_positive: while is_neg.any() {
            reflected = z.le(Vf64::<S>::splat(-20.0));

            let mut refl_res = unsafe { Vf64::<S>::undefined() };

            // sine is expensive, so branch for it.
            if unlikely!(reflected.any()) {
                // TODO: Improve error around integers
                refl_res = z * Self::sin::<P>(z * pi);

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
                mod_z = is_neg.select(mod_z + one, mod_z);
                is_neg = mod_z.is_negative();
            }

            z = reflected.select(-z, mod_z);
            res = reflected.select(refl_res, res);

            break 'goto_positive;
        }

        // label
        //positive:

        // Integers

        let zf = z.floor();
        let z_int = zf.eq(z);
        let mut fact_res = one;

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
            fact_res = is_neg.select(Vf64::<S>::nan(), fact_res);
            // approaching zero from either side results in +/- infinity
            fact_res = orig_z.eq(zero).select(Vf64::<S>::infinity().copysign(orig_z), fact_res);

            if bitmask.all() {
                return fact_res;
            }
        }

        // Tiny

        let sqrt_epsilon = Vf64::<S>::splat(0.1490116119384765625e-7);
        let euler = Vf64::<S>::splat(5.772156649015328606065120900824024310e-01);
        let tiny = z.lt(sqrt_epsilon);
        let tiny_res = one / z - euler;

        // Full

        let n00 = Vf64::<S>::splat(23531376880.41075968857200767445163675473);
        let n01 = Vf64::<S>::splat(42919803642.64909876895789904700198885093);
        let n02 = Vf64::<S>::splat(35711959237.35566804944018545154716670596);
        let n03 = Vf64::<S>::splat(17921034426.03720969991975575445893111267);
        let n04 = Vf64::<S>::splat(6039542586.352028005064291644307297921070);
        let n05 = Vf64::<S>::splat(1439720407.311721673663223072794912393972);
        let n06 = Vf64::<S>::splat(248874557.8620541565114603864132294232163);
        let n07 = Vf64::<S>::splat(31426415.58540019438061423162831820536287);
        let n08 = Vf64::<S>::splat(2876370.628935372441225409051620849613599);
        let n09 = Vf64::<S>::splat(186056.2653952234950402949897160456992822);
        let n10 = Vf64::<S>::splat(8071.672002365816210638002902272250613822);
        let n11 = Vf64::<S>::splat(210.8242777515793458725097339207133627117);
        let n12 = Vf64::<S>::splat(2.506628274631000270164908177133837338626);

        let d01 = Vf64::<S>::splat(39916800.0);
        let d02 = Vf64::<S>::splat(120543840.0);
        let d03 = Vf64::<S>::splat(150917976.0);
        let d04 = Vf64::<S>::splat(105258076.0);
        let d05 = Vf64::<S>::splat(45995730.0);
        let d06 = Vf64::<S>::splat(13339535.0);
        let d07 = Vf64::<S>::splat(2637558.0);
        let d08 = Vf64::<S>::splat(357423.0);
        let d09 = Vf64::<S>::splat(32670.0);
        let d10 = Vf64::<S>::splat(1925.0);
        let d11 = Vf64::<S>::splat(66.0);

        let gh = Vf64::<S>::splat(6.024680040776729583740234375 - 0.5);

        let z2 = z * z;
        let z4 = z2 * z2;
        let z8 = z4 * z4;

        let lanczos_sum = poly_12(
            z, z2, z4, z8, n00, n01, n02, n03, n04, n05, n06, n07, n08, n09, n10, n11, n12,
        ) / poly_12(
            z, z2, z4, z8, zero, d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, one,
        );

        let zgh = z + gh;
        let lzgh = Self::ln::<P>(zgh);

        // (z * lzfg) > ln(f64::MAX)
        let very_large = (z * lzgh).gt(Vf64::<S>::splat(
            709.78271289338399672769243071670056097572649130589734950577761613,
        ));

        // only compute powf once
        let h = Self::powf::<P>(zgh, very_large.select(z.mul_sub(half, quarter), z - half));

        let normal_res = lanczos_sum * very_large.select(h * h, h) / Self::exp::<P>(zgh);

        res *= tiny.select(tiny_res, normal_res);

        reflected.select(-pi / res, z_int.select(fact_res, res))
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
    Vi32::<S>::from_bits(<Vu32<S> as SimdFromCast<S, Vu64<S>>>::from_cast(
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
fn ln_d_internal<S: Simd, P: Policy>(x0: Vf64<S>, p1: bool) -> Vf64<S> {
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
    let mut fe = <Vf64<S> as SimdFromCast<S, Vi32<S>>>::from_cast(exponent::<S>(x1));

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

    if !P::POLICY.check_overflow() {
        return res;
    }

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
fn atan_internal<S: Simd, P: Policy>(y: Vf64<S>, x: Vf64<S>, atan2: bool) -> Vf64<S> {
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

        if P::POLICY.check_overflow() {
            let both_inf = x.is_infinite() & y.is_infinite();

            // TODO: Benchmark this branch
            if unlikely!(both_inf.any()) {
                x2 = both_inf.select(x2 & neg_one, x2);
                y2 = both_inf.select(y2 & neg_one, y2);
            }
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
        re = x.select_negative(Vf64::<S>::splat(PI) - re, re);
    }

    re.combine_sign(y)
}

#[inline(always)]
fn asin_internal<S: Simd, P: Policy>(x: Vf64<S>, acos: bool) -> Vf64<S> {
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

        px = poly_5(x1, x2, x4, p0asin, p1asin, p2asin, p3asin, p4asin, p5asin);
        qx = poly_5(x1, x2, x4, q0asin, q1asin, q2asin, q3asin, q4asin, one);
    }

    // if any are big
    if bitmask.any() {
        let r4asin = Vf64::<S>::splat(2.967721961301243206100E-3);
        let r3asin = Vf64::<S>::splat(-5.634242780008963776856E-1);
        let r2asin = Vf64::<S>::splat(6.968710824104713396794E0);
        let r1asin = Vf64::<S>::splat(-2.556901049652824852289E1);
        let r0asin = Vf64::<S>::splat(2.853665548261061424989E1);
        let s3asin = Vf64::<S>::splat(-2.194779531642920639778E1);
        let s2asin = Vf64::<S>::splat(1.470656354026814941758E2);
        let s1asin = Vf64::<S>::splat(-3.838770957603691357202E2);
        let s0asin = Vf64::<S>::splat(3.424398657913078477438E2);

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
        let z1 = x.select_negative(Vf64::<S>::splat(PI) - z1, z1);
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
fn exp_d_internal<S: Simd, P: Policy>(x0: Vf64<S>, mode: ExpMode) -> Vf64<S> {
    let zero = Vf64::<S>::zero();
    let one = Vf64::<S>::one();

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

    // Taylor coefficients, 1/n!
    // Not using minimax approximation because we prioritize precision close to x = 0
    let mut z = x.poly(&[
        0.0,
        1.0 / 1.0,
        1.0 / 2.0,
        1.0 / 6.0,
        1.0 / 24.0,
        1.0 / 120.0,
        1.0 / 720.0,
        1.0 / 5040.0,
        1.0 / 40320.0,
        1.0 / 362880.0,
        1.0 / 3628800.0,
        1.0 / 39916800.0,
        1.0 / 479001600.0,
        1.0 / 6227020800.0,
    ]);

    let n2 = pow2n_d::<S>(r);

    if mode == ExpMode::Expm1 {
        z = z.mul_add(n2, n2 - one);
    } else {
        z = z.mul_add(n2, n2); // (z + 1.0f) * n2
    }

    let in_range = x0.abs().lt(Vf64::<S>::splat(max_x)) & x0.is_finite();

    if likely!(in_range.all()) {
        return z;
    }

    let underflow_value = if mode == ExpMode::Expm1 {
        Vf64::<S>::neg_one()
    } else {
        Vf64::<S>::zero()
    };

    if P::POLICY.check_overflow() {
        r = x0.select_negative(underflow_value, Vf64::<S>::infinity());
        z = in_range.select(z, r);
        z = x0.is_nan().select(x0, z);
    }

    z
}
