use super::{common::*, *};

impl<S: Simd> SimdVectorizedMathInternal<S> for f64
where
    <S as Simd>::Vf64: SimdFloatVector<S, Element = f64>,
{
    type Vf = <S as Simd>::Vf64;

    fn sin_cos(xx: Self::Vf) -> (Self::Vf, Self::Vf) {
        unimplemented!()
    }

    fn sinh(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }

    fn tanh(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn asin(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn acos(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn atan(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn atan2(y: Self::Vf, x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn asinh(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn acosh(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn atanh(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn exp(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn exph(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn exp2(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn exp10(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn powf(x: Self::Vf, e: Self::Vf) -> Self::Vf {
        unimplemented!()
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
        x.ln() * Vf64::<S>::splat(std::f64::consts::LOG2_E)
    }

    #[inline(always)]
    fn log10(x: Self::Vf) -> Self::Vf {
        x.ln() * Vf64::<S>::splat(std::f64::consts::LOG10_E)
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

    fn erfinv(y: Self::Vf) -> Self::Vf {
        let one = Vf64::<S>::one();

        let a = y.abs();

        let w = -a.nmul_add(a, one).ln();

        let mut p0 = {
            // https://www.desmos.com/calculator/06q98crjp0
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

    let blend = x.gt(Vf64::<S>::splat(std::f64::consts::SQRT_2 * 0.5));

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
