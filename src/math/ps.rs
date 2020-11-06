use super::{common::*, *};

impl<S: Simd> SimdVectorizedMathInternal<S> for f32
where
    <S as Simd>::Vf32: SimdFloatVector<S, Element = f32>,
{
    type Vf = <S as Simd>::Vf32;

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

        let y: Vf32<S> = (xa * Vf32::<S>::splat(2.0 / std::f32::consts::PI)).round();
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
        let r0 = Vf32::<S>::splat(1.66667160211E-1);
        let r1 = Vf32::<S>::splat(8.33028376239E-3);
        let r2 = Vf32::<S>::splat(2.03721912945E-4);

        let x = x0.abs();

        let x_small = x.le(Vf32::<S>::one());

        let mut y1 = unsafe { Vf32::<S>::undefined() };
        let mut y2 = unsafe { Vf32::<S>::undefined() };

        // use bitmask directly to avoid two calls
        let bitmask = x_small.bitmask();

        // if any are small
        if bitmask != 0 {
            let x2 = x * x;
            y1 = poly_2(x2, x2 * x2, r0, r1, r2).mul_add(x2 * x, x);
        }

        // if not all are small
        if bitmask != Mask::<S, Vf32<S>>::FULL_BITMASK {
            y2 = x.exph();
            y2 -= Vf32::<S>::splat(0.25) / y2;
        }

        x_small.select(y1, y2).combine_sign(x0)
    }

    #[inline(always)]
    fn tanh(x0: Self::Vf) -> Self::Vf {
        let r0 = Vf32::<S>::splat(-3.33332819422E-1);
        let r1 = Vf32::<S>::splat(1.33314422036E-1);
        let r2 = Vf32::<S>::splat(-5.37397155531E-2);
        let r3 = Vf32::<S>::splat(2.06390887954E-2);
        let r4 = Vf32::<S>::splat(-5.70498872745E-3);

        let x = x0.abs();
        let x_small = x.le(Vf32::<S>::splat(0.625));

        let mut y1 = unsafe { Vf32::<S>::undefined() };
        let mut y2 = unsafe { Vf32::<S>::undefined() };

        // use bitmask directly to avoid two calls
        let bitmask = x_small.bitmask();

        // if any are small
        if bitmask != 0 {
            let x2 = x * x;
            let x4 = x2 * x2;

            y1 = poly_4(x2, x4, x4 * x4, r0, r1, r2, r3, r4).mul_add(x2 * x, x);
        }

        // if not all are small
        if bitmask != Mask::<S, Vf32<S>>::FULL_BITMASK {
            y2 = (x + x).exp();
            y2 = Vf32::<S>::one() - Vf32::<S>::splat(2.0) / (y2 + Vf32::<S>::one());
        }

        let x_big = x.gt(Vf32::<S>::splat(44.4));

        y1 = x_small.select(y1, y2);
        y1 = x_big.select(Vf32::<S>::one(), y1);

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

        let not_small = t.ge(Vf32::<S>::splat(std::f32::consts::SQRT_2 - 1.0)); // t >= tan  pi/8
        let not_big = t.le(Vf32::<S>::splat(std::f32::consts::SQRT_2 + 1.0)); // t <= tan 3pi/8

        let s = not_big.select(
            Vf32::<S>::splat(std::f32::consts::FRAC_PI_4),
            Vf32::<S>::splat(std::f32::consts::FRAC_PI_2),
        ) & not_small.value(); // select(not_small, s, 0.0);

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
        let not_small = t.ge(Vf32::<S>::splat(std::f32::consts::SQRT_2 - 1.0));

        let a = t + (not_small.value() & neg_one);
        let b = Vf32::<S>::one() + (not_small.value() & t);

        let s = not_small.value() & Vf32::<S>::splat(std::f32::consts::FRAC_PI_4);

        let z = a / b;
        let z2 = z * z;

        let mut re = poly_3(z2, z2 * z2, p0atanf, p1atanf, p2atanf, p3atanf).mul_add(z2 * z, z + s);

        re = swap_xy.select(Vf32::<S>::splat(std::f32::consts::FRAC_PI_2) - re, re);
        re = (x | y).eq(zero).select(zero, re); // atan2(0,+0) = 0 by convention
        re = x.is_negative().select(Vf32::<S>::splat(std::f32::consts::PI) - re, re); // also for x = -0.

        re
    }

    #[inline(always)]
    fn asinh(x0: Self::Vf) -> Self::Vf {
        let r0 = Vf32::<S>::splat(-1.6666288134E-1);
        let r1 = Vf32::<S>::splat(7.4847586088E-2);
        let r2 = Vf32::<S>::splat(-4.2699340972E-2);
        let r3 = Vf32::<S>::splat(2.0122003309E-2);

        let x = x0.abs();
        let x2 = x0 * x0;

        let x_small = x.le(Vf32::<S>::splat(0.51));
        let x_huge = x.gt(Vf32::<S>::splat(1e10));

        let mut y1 = unsafe { Vf32::<S>::undefined() };
        let mut y2 = unsafe { Vf32::<S>::undefined() };

        let bitmask = x_small.bitmask();

        if bitmask != 0 {
            y1 = poly_3(x2, x2 * x2, r0, r1, r2, r3).mul_add(x2 * x, x);
        }

        if bitmask != Mask::<S, Vf32<S>>::FULL_BITMASK {
            y2 = ((x2 + Vf32::<S>::one()).sqrt() + x).ln();

            if unlikely!(x_huge.any()) {
                y2 = x_huge.select(x.ln() + Vf32::<S>::splat(std::f32::consts::LN_2), y2);
            }
        }

        x_small.select(y1, y2).combine_sign(x0)
    }

    #[inline(always)]
    fn acosh(x0: Self::Vf) -> Self::Vf {
        let r0 = Vf32::<S>::splat(1.4142135263E0);
        let r1 = Vf32::<S>::splat(-1.1784741703E-1);
        let r2 = Vf32::<S>::splat(2.6454905019E-2);
        let r3 = Vf32::<S>::splat(-7.5272886713E-3);
        let r4 = Vf32::<S>::splat(1.7596881071E-3);

        let one = Vf32::<S>::one();

        let x1 = x0 - one;

        let undef = x0.lt(one); // result is NAN
        let x_small = x1.lt(Vf32::<S>::splat(0.49)); // use Pade approximation if abs(x-1) < 0.5
        let x_huge = x1.gt(Vf32::<S>::splat(1e10));

        let mut y1 = unsafe { Vf32::<S>::undefined() };
        let mut y2 = unsafe { Vf32::<S>::undefined() };

        let bitmask = x_small.bitmask();

        // if any are small
        if bitmask != 0 {
            let x2 = x1 * x1;
            let x4 = x2 * x2;
            y1 = x1.sqrt() * poly_4(x1, x2, x4, r0, r1, r2, r3, r4);
            y1 = undef.select(Vf32::<S>::nan(), y1);
        }

        // if not all are small
        if bitmask != Mask::<S, Vf32<S>>::FULL_BITMASK {
            y2 = (x0.mul_sub(x0, one).sqrt() + x0).ln();

            if x_huge.any() {
                y2 = x_huge.select(x0.ln() + Vf32::<S>::splat(std::f32::consts::LN_2), y2);
            }
        }

        x_small.select(y1, y2)
    }

    #[inline(always)]
    fn atanh(x0: Self::Vf) -> Self::Vf {
        let r0 = Vf32::<S>::splat(3.33337300303E-1);
        let r1 = Vf32::<S>::splat(1.99782164500E-1);
        let r2 = Vf32::<S>::splat(1.46691431730E-1);
        let r3 = Vf32::<S>::splat(8.24370301058E-2);
        let r4 = Vf32::<S>::splat(1.81740078349E-1);

        let x = x0.abs();

        let x_small = x.lt(Vf32::<S>::splat(0.5));

        let mut y1 = unsafe { Vf32::<S>::undefined() };
        let mut y2 = unsafe { Vf32::<S>::undefined() };

        let bitmask = x_small.bitmask();

        if bitmask != 0 {
            let x2 = x * x;
            let x4 = x2 * x2;
            let x8 = x4 * x4;

            y1 = poly_4(x2, x4, x8, r0, r1, r2, r3, r4).mul_add(x2 * x, x);
        }

        if bitmask != Mask::<S, Vf32<S>>::FULL_BITMASK {
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

    fn powf(x: Self::Vf, e: Self::Vf) -> Self::Vf {
        unimplemented!()
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
        x.ln() * Vf32::<S>::splat(std::f32::consts::LOG2_E)
    }

    #[inline(always)]
    fn log10(x: Self::Vf) -> Self::Vf {
        x.ln() * Vf32::<S>::splat(std::f32::consts::LOG10_E)
    }

    #[inline(always)]
    fn erf(x: Self::Vf) -> Self::Vf {
        /* Abramowitz and Stegun, 7.1.28. */
        let a0 = Vf32::<S>::one();
        let a1 = Vf32::<S>::splat(0.0705230784);
        let a2 = Vf32::<S>::splat(0.0422820123);
        let a3 = Vf32::<S>::splat(0.0092705272);
        let a4 = Vf32::<S>::splat(0.0001520143);
        let a5 = Vf32::<S>::splat(0.0002765672);
        let a6 = Vf32::<S>::splat(0.0000430638);

        let b = a0 - (a0 - x.abs()); // crush denormals
        let b2 = b * b;
        let b4 = b2 * b2;

        let r = poly_6(b, b2, b4, a0, a1, a2, a3, a4, a5, a6);

        let r2 = r * r;
        let r4 = r2 * r2;
        let r8 = r4 * r4;
        let r16 = r8 * r8;

        (a0 - a0 / r16).copysign(x)
    }

    #[inline(always)]
    fn erfinv(x: Self::Vf) -> Self::Vf {
        /*
            Approximating the erfinv function, Mike Giles
            https://people.maths.ox.ac.uk/gilesm/files/gems_erfinv.pdf
        */
        let one = Vf32::<S>::one();

        let a = x.abs();

        let w = -((one - a) * (one + a)).ln();

        let p0 = {
            let p0low = Vf32::<S>::splat(1.50140941);
            let p1low = Vf32::<S>::splat(0.246640727);
            let p2low = Vf32::<S>::splat(-0.00417768164);
            let p3low = Vf32::<S>::splat(-0.00125372503);
            let p4low = Vf32::<S>::splat(0.00021858087);
            let p5low = Vf32::<S>::splat(-4.39150654e-06);
            let p6low = Vf32::<S>::splat(-3.5233877e-06);
            let p7low = Vf32::<S>::splat(3.43273939e-07);
            let p8low = Vf32::<S>::splat(2.81022636e-08);

            let w1 = w - Vf32::<S>::splat(2.5);
            let w2 = w1 * w1;
            let w4 = w2 * w2;
            let w8 = w4 * w4;

            poly_8(
                w1, w2, w4, w8, p0low, p1low, p2low, p3low, p4low, p5low, p6low, p7low, p8low,
            )
        };

        let mut p1 = unsafe { Vf32::<S>::undefined() };

        let w_big = w.ge(Vf32::<S>::splat(5.0)); // at around |x| > 0.99662533231, so unlikely

        // avoids a costly sqrt and polynomial if false
        if unlikely!(w_big.any()) {
            let p0high = Vf32::<S>::splat(2.83297682);
            let p1high = Vf32::<S>::splat(1.00167406);
            let p2high = Vf32::<S>::splat(0.00943887047);
            let p3high = Vf32::<S>::splat(-0.0076224613);
            let p4high = Vf32::<S>::splat(0.00573950773);
            let p5high = Vf32::<S>::splat(-0.00367342844);
            let p6high = Vf32::<S>::splat(0.00134934322);
            let p7high = Vf32::<S>::splat(0.000100950558);
            let p8high = Vf32::<S>::splat(-0.000200214257);

            let w1 = w.sqrt() - Vf32::<S>::splat(3.0);
            let w2 = w1 * w1;
            let w4 = w2 * w2;
            let w8 = w4 * w4;

            p1 = poly_8(
                w1, w2, w4, w8, p0high, p1high, p2high, p3high, p4high, p5high, p6high, p7high, p8high,
            );

            p1 = a.eq(one).select(Vf32::<S>::infinity(), p1); // erfi(x == 1) = inf
            p1 = a.gt(one).select(Vf32::<S>::nan(), p1); // erfi(x > 1) = NaN
        }

        w_big.select(p1, p0) * x
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
    use std::f32::consts::{LN_10, LN_2, LOG10_2, LOG2_E};

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

    if mode == ExpMode::Exph {
        r -= Vf32::<S>::one();
    }

    let n2 = pow2n_f::<S>(r);

    if mode == ExpMode::Expm1 {
        z = z.mul_add(n2, n2 - Vf32::<S>::one());
    } else {
        z = (z + Vf32::<S>::one()) * n2;
    }

    let in_range = x0.abs().lt(Vf32::<S>::splat(max_x)) & x0.is_finite().cast_to();

    if unlikely!(!in_range.all()) {
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
    }

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
        let z1 = x.is_positive().select(Vf32::<S>::splat(std::f32::consts::PI) - z1, z1);
        let z2 = Vf32::<S>::splat(std::f32::consts::FRAC_PI_2) - z.combine_sign(x);

        is_big.select(z1, z2)
    } else {
        let z1 = Vf32::<S>::splat(std::f32::consts::FRAC_PI_2) - z1;

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

    let x1 = if p1 { x0 + Vf32::<S>::one() } else { x0 };

    let mut x = fraction2::<S>(x1);
    let mut e = exponent::<S>(x1);

    let blend = x.gt(Vf32::<S>::splat(std::f32::consts::SQRT_2 * 0.5));

    x += !blend.value() & x; // conditional addition
    e += Vi32::<S>::from_bits(blend.value().into_bits() & Vu32::<S>::one()); // conditional (signed) addition

    // TODO: Fix this cast when the type inference bug hits stable
    let fe = <Vf32<S> as SimdCastFrom<S, Vi32<S>>>::from_cast(e);

    let xp1 = x - Vf32::<S>::one();

    if p1 {
        // log(x+1). Avoid loss of precision when adding 1 and later subtracting 1 if exponent = 0
        x = e.eq(Vi32::<S>::zero()).select(x0, xp1);
    } else {
        // log(x). Expand around 1.0
        x = xp1;
    }

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
