use super::{common::*, *};

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

    unimplemented!()
}

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

        let mut overflow = q.gt(Vu32::<S>::splat(0x2000000)); // q big if overflow
        overflow &= xa.is_finite().cast_to();

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

    fn asinh(x0: Self::Vf) -> Self::Vf {
        let r0 = Vf32::<S>::splat(-1.6666288134E-1);
        let r1 = Vf32::<S>::splat(7.4847586088E-2);
        let r2 = Vf32::<S>::splat(-4.2699340972E-2);
        let r3 = Vf32::<S>::splat(2.0122003309E-2);

        let x = x0.abs();
        let x2 = x0 * x0;

        let x_small = x.le(Vf32::<S>::splat(0.5));
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

    fn acosh(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn atanh(x: Self::Vf) -> Self::Vf {
        unimplemented!()
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
    fn ln(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn ln_1p(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn log2(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn log10(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn erf(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
    fn ierf(x: Self::Vf) -> Self::Vf {
        unimplemented!()
    }
}
