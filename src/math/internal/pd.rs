use crate::math::{consts::FloatConsts as _, policy::policies::ExtraPrecision};
use core::f64::consts::{FRAC_1_PI, LN_10, LOG2_E, SQRT_2};

use super::*;

impl<R> MathInternal<f64> for R
where
    R: FloatRegister<Element = f64>,
{
    #[inline(always)]
    fn sincos<P: Policy>(xx: Vf<Self>) -> (Vf<Self>, Vf<Self>) {
        let dp1 = Vf::splat(7.853981554508209228515625E-1 * 2.0);
        let dp2 = Vf::splat(7.94662735614792836714E-9 * 2.0);
        let dp3 = Vf::splat(3.06161699786838294307E-17 * 2.0);
        let xa = xx.abs();

        let y = (xa * Vf::FRAC_2_PI).round();
        let q = Vu::<R>::fast_from(y);
        //let q = unsafe { y.to_uint_fast() };

        // Reduce by extended precision modular arithmetic
        // x = ((xa - y * DP1F) - y * DP2F) - y * DP3F;
        let x = y.nmul_add(dp3, y.nmul_add(dp2, y.nmul_add(dp1, xa)));

        // Taylor expansion of sin and cos, valid for -pi/4 <= x <= pi/4
        let x2 = x * x;
        let x4 = x2 * x2;

        let mut s = x2.poly_p::<P, 6>(&[
            -1.66666666666666307295E-1,
            8.33333333332211858878E-3,
            -1.98412698295895385996E-4,
            2.75573136213857245213E-6,
            -2.50507477628578072866E-8,
            1.58962301576546568060E-10,
        ]);

        let mut c = x2.poly_p::<P, 6>(&[
            4.16666666666665929218E-2,
            -1.38888888888730564116E-3,
            2.48015872888517045348E-5,
            -2.75573141792967388112E-7,
            2.08757008419747316778E-9,
            -1.13585365213876817300E-11,
        ]);

        s = s.mul_adde(x2 * x, x); // s = x + (x * x2) * s;
        c = c.mul_adde(x4, x2.nmul_adde(Vf::HALF, Vf::ONE)); // c = 1.0 - x2 * 0.5 + (x2 * x2) * c;

        // swap sin and cos if odd quadrant
        let swap = (q & Vu::<R>::ONE).cmp_ne(Vu::<R>::ZERO);

        if P::POLICY.check_overflow {
            let overflow = y.cmp_gt(Vf::splat((1u64 << 52) as f64 - 1.0)) & xa.is_finite();

            let s = overflow.select(Vf::ZERO, s);
            let c = overflow.select(Vf::ONE, c);
        }

        let sin1 = swap.select(c, s);
        let cos1 = swap.select(s, c);

        let signsin = Vf::from_bits(q << 62) ^ xx;
        let signcos = Vf::from_bits(((q + Vu::<R>::ONE) & Vu::<R>::splat(2)) << 62);

        // combine signs
        (sin1.mul_sign(signsin), cos1 ^ signcos)
    }

    #[inline(always)]
    fn sinh<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    #[inline(always)]
    fn cosh<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    #[inline(always)]
    fn tanh<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    #[inline(always)]
    fn asin<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    #[inline(always)]
    fn acos<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    #[inline(always)]
    fn atan<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    #[inline(always)]
    fn atan2<P: Policy>(y: Vf<Self>, x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    #[inline(always)]
    fn asinh<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    #[inline(always)]
    fn acosh<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    #[inline(always)]
    fn atanh<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    #[inline(always)]
    fn exp<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        exp_d_internal::<Self, P, EXP_MODE_EXP>(x)
    }

    #[inline(always)]
    fn exph<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        exp_d_internal::<Self, P, EXP_MODE_EXPH>(x)
    }

    #[inline(always)]
    fn exp2<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        exp_d_internal::<Self, P, EXP_MODE_POW2>(x)
    }

    #[inline(always)]
    fn exp10<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        exp_d_internal::<Self, P, EXP_MODE_POW10>(x)
    }

    #[inline(always)]
    fn exp_m1<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        exp_d_internal::<Self, P, EXP_MODE_EXPM1>(x)
    }

    #[inline(always)]
    fn powf<P: Policy>(x: Vf<Self>, e: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    #[inline(always)]
    fn cbrt<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    #[inline(always)]
    fn ln<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        ln_d_internal::<Self, P, false>(x)
    }

    #[inline(always)]
    fn ln1p<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        ln_d_internal::<Self, P, true>(x)
    }

    #[inline(always)]
    fn log2<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        ln_d_internal::<Self, P, false>(x) * Vf::LOG2_E
    }

    #[inline(always)]
    fn log10<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        ln_d_internal::<Self, P, false>(x) * Vf::LOG10_2
    }

    #[inline(always)]
    fn erf<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    #[inline(always)]
    fn erfc<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    #[inline(always)]
    fn erfinv<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }
}

#[inline(always)]
fn fraction2<R: MathInternal<f64>>(x: Vf<R>) -> Vf<R> {
    // set exponent to 0 + bias
    (x & Vf::splat(f64::from_bits(0x000FFFFFFFFFFFFF))) | Vf::splat(f64::from_bits(0x3FE0000000000000))
}

#[inline(always)]
fn exponent<R: MathInternal<f64>>(x: Vf<R>) -> Vs<R> {
    // shift out sign, extract exp, subtract bias
    Vs::<R>::from_bits((Vu::<R>::from_bits(x) << 1) >> 53) - Vs::<R>::splat(0x3FF)
}

#[inline(always)]
fn exponent_f<R: MathInternal<f64>>(x: Vf<R>) -> Vf<R> {
    let pow2_52 = Vf::<R>::splat(4503599627370496.0);
    let bias = Vf::<R>::splat(1023.0);

    Vf::from_bits((Vu::<R>::from_bits(x) >> 52) | pow2_52.into_bits()) - (pow2_52 + bias)
}

#[inline(always)]
fn ln_d_internal<R: MathInternal<f64>, P: Policy, const P1: bool>(x0: Vf<R>) -> Vf<R> {
    let ln2_hi = Vf::splat(0.693359375);
    let ln2_lo = Vf::splat(-2.121944400546905827679E-4);
    let x1 = if P1 { x0 + Vf::ONE } else { x0 };

    let mut x = fraction2::<R>(x1);
    let mut fe = Vf::from(exponent::<R>(x1));

    let blend = x.cmp_gt(Vf::splat(SQRT_2 * 0.5));

    x = blend.select(x, x + x); // x = x.conditional_add(x, !blend);
    fe = blend.select(fe + Vf::ONE, fe); // fe = fe.conditional_add(Vf::ONE, blend);

    let xp1 = x - Vf::ONE;

    x = if P1 {
        // log(x+1). Avoid loss of precision when adding 1 and later subtracting 1 if exponent = 0
        fe.cmp_eq(Vf::ZERO).select(x0, xp1)
    } else {
        // log(x). Expand around 1.0
        xp1
    };

    let x2 = x * x;
    let x3 = x * x2;

    let mut res =
        x3 * x.poly_p::<P, 6>(&[
            7.70838733755885391666E0,
            1.79368678507819816313E1,
            1.44989225341610930846E1,
            4.70579119878881725854E0,
            4.97494994976747001425E-1,
            1.01875663804580931796E-4,
        ]) / x.poly_p::<P, 6>(&[
            2.31251620126765340583E1,
            7.11544750618563894466E1,
            8.29875266912776603211E1,
            4.52279145837532221105E1,
            1.12873587189167450590E1,
            1.0,
        ]);

    res = fe.mul_adde(ln2_lo, res); // res += fe * ln2_lo;
    res += x2.nmul_adde(Vf::HALF, x); // res += x - 0.5 * x2;
    res = fe.mul_adde(ln2_hi, res); // res += fe * ln2_hi;

    if !P::POLICY.check_overflow {
        return res;
    }

    let overflow = !x1.is_finite();
    let underflow = x1.cmp_lt(Vf::splat(2.2250738585072014E-308));

    if crate::likely((overflow | underflow).none()) {
        return res;
    }

    res = underflow.select(Vf::NAN, res); // x1 < 0 gives NAN
    res = x1.is_zero_or_subnormal().select(Vf::NEG_INFINITY, res); // x1 == 0 gives -INF
    res = overflow.select(x1, res); // INF or NAN goes through
    res = (x1.is_infinite() & x1.is_negative()).select(Vf::NAN, res); // -INF gives NAN

    res
}

#[inline(always)]
fn atan_internal<R: MathInternal<f64>, P: Policy, const ATAN2: bool>(y: Vf<R>, x: Vf<R>) -> Vf<R> {
    let morebits = Vf::splat(6.123233995736765886130E-17);
    let morebitso2 = Vf::splat(6.123233995736765886130E-17 * 0.5);
    let t3po8 = Vf::splat(SQRT_2 + 1.0);

    let mut swapxy = Mask::FALSY;

    let t = if ATAN2 {
        let x1 = x.abs();
        let y1 = y.abs();

        swapxy = y1.cmp_gt(x1);

        let mut x2 = swapxy.select(y1, x1);
        let mut y2 = swapxy.select(x1, y1);

        if P::POLICY.check_overflow {
            let both_inf = x.is_infinite() & y.is_infinite();

            // TODO: Benchmark this branch
            if crate::unlikely(both_inf.any()) {
                x2 = both_inf.select(x2 & Vf::NEG_ONE, x2);
                y2 = both_inf.select(y2 & Vf::NEG_ONE, y2);
            }
        }

        y2 / x2
    } else {
        y.abs()
    };

    let not_big = t.cmp_le(t3po8);
    let not_small = t.cmp_ge(Vf::splat(0.66));

    let s = not_big.select(Vf::FRAC_PI_4, Vf::FRAC_PI_2) & not_small.value();

    let fac = not_big.select(morebitso2, morebits) & not_small.value();

    let a = (not_big.value() & t) + (not_small.value() & Vf::NEG_ONE);
    let b = (not_big.value() & Vf::ONE) + (not_small.value() & t);

    let z = a / b;

    let zz = z * z;

    let re0 = zz.poly_p::<P, 5>(&[
        -6.485021904942025371773E1,
        -1.228866684490136173410E2,
        -7.500855792314704667340E1,
        -1.615753718733365076637E1,
        -8.750608600031904122785E-1,
    ]) / zz.poly_p::<P, 6>(&[
        1.945506571482613964425E2,
        4.853903996359136964868E2,
        4.328810604912902668951E2,
        1.650270098316988542046E2,
        2.485846490142306297962E1,
        1.0,
    ]);

    // place additions before mul_add to lessen dependency chain
    let mut re = re0.mul_adde(z * zz, z + s + fac);

    if ATAN2 {
        re = swapxy.select(Vf::FRAC_PI_2 - re, re);
        re = (x | y).cmp_eq(Vf::ZERO).select(Vf::ZERO, re); // atan2(0,0) = 0 by convention
        // also for x = -0.
        re = x.select_negative(Vf::PI - re, re);
    }

    re.mul_sign(y)
}

#[inline(always)]
fn asin_internal<R: MathInternal<f64>, P: Policy, const ACOS: bool>(x: Vf<R>) -> Vf<R> {
    let xa = x.abs();

    let is_big = xa.cmp_ge(Vf::splat(0.625));

    let x1 = is_big.select(Vf::ONE - xa, xa * xa);

    let x2 = x1 * x1;
    let x4 = x2 * x2;
    let x8 = x4 * x4;

    let undef = Vf::EMPTY;

    let mut px = undef;
    let mut qx = undef;
    let mut rx = undef;
    let mut sx = undef;
    let mut xb = undef;

    // if not all are big (if any are small)
    if P::POLICY.avoid_branching || !is_big.all() {
        px = x1.poly_p::<P, 6>(&[
            -8.198089802484824371615E0,
            1.956261983317594739197E1,
            -1.626247967210700244449E1,
            5.444622390564711410273E0,
            -6.019598008014123785661E-1,
            4.253011369004428248960E-3,
        ]);

        qx = x1.poly_p::<P, 6>(&[
            -4.918853881490881290097E1,
            1.395105614657485689735E2,
            -1.471791292232726029859E2,
            7.049610280856842141659E1,
            -1.474091372988853791896E1,
            1.0,
        ]);
    }

    // if any are big
    if P::POLICY.avoid_branching || is_big.any() {
        xb = (x1 + x1).sqrt();

        rx = x1.poly_p::<P, 5>(&[
            2.853665548261061424989E1,
            -2.556901049652824852289E1,
            6.968710824104713396794E0,
            -5.634242780008963776856E-1,
            2.967721961301243206100E-3,
        ]);

        sx = x1.poly_p::<P, 5>(&[
            3.424398657913078477438E2,
            -3.838770957603691357202E2,
            1.470656354026814941758E2,
            -2.194779531642920639778E1,
            1.0,
        ]);
    }

    let vx = is_big.select(rx, px);
    let wx = is_big.select(sx, qx);

    let y1 = vx / wx * x1;

    // avoid branching again for this single instruction, just do it
    let z1 = xb.mul_adde(y1, xb);
    let z2 = xa.mul_adde(y1, xa);

    if ACOS {
        let z1 = x.select_negative(Vf::PI - z1, z1);
        let z2 = Vf::FRAC_PI_2 - z2.mul_sign(x);
        is_big.select(z1, z2)
    } else {
        let z1 = Vf::FRAC_PI_2 - z1;
        is_big.select(z1, z2).mul_sign(x)
    }
}

#[inline(always)]
fn pow2n_d<R: MathInternal<f64>>(n: Vf<R>) -> Vf<R> {
    let pow2_52 = Vf::splat(4503599627370496.0);
    let bias = Vf::splat(1023.0);

    (n + (bias + pow2_52)) << 52
}

#[inline(always)]
fn exp_d_internal<R: MathInternal<f64>, P: Policy, const MODE: u8>(x0: Vf<R>) -> Vf<R> {
    let mut x = x0;
    let mut r;

    let max_x;

    match MODE {
        EXP_MODE_POW2 => {
            max_x = 1022.0;

            r = x0.round();

            x -= r;
            x *= Vf::LN_2;
        }
        EXP_MODE_POW10 => {
            max_x = 307.65;

            let log10_2_hi = Vf::splat(0.30102999554947019); // log10(2) in two parts
            let log10_2_lo = Vf::splat(1.1451100899212592E-10);

            r = (x0 * Vf::splat(LN_10 * LOG2_E)).round();

            x = r.nmul_adde(log10_2_hi, x); // x -= r * log10_2_hi;
            x = r.nmul_adde(log10_2_lo, x); // x -= r * log10_2_lo;
            x *= Vf::LN_10;
        }
        _ => {
            max_x = const { if MODE == EXP_MODE_EXP { 708.39 } else { 709.7 } };

            let ln2d_hi = Vf::splat(0.693145751953125);
            let ln2d_lo = Vf::splat(1.42860682030941723212E-6);

            r = (x0 * Vf::splat(LOG2_E)).round();

            x = r.nmul_adde(ln2d_hi, x); // x -= r * ln2_hi;
            x = r.nmul_adde(ln2d_lo, x); // x -= r * ln2_lo;

            if MODE == EXP_MODE_EXPH {
                r -= Vf::ONE;
            }
        }
    }

    // Taylor coefficients, 1/n!
    // Not using minimax approximation because we prioritize precision close to x = 0
    let mut z = x.poly_p::<P, 14>(&[
        0.0,
        1.0,
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

    let n2 = pow2n_d::<R>(r);

    z = match MODE {
        EXP_MODE_EXPM1 => z.mul_adde(n2, n2 - Vf::ONE),
        _ => z.mul_adde(n2, n2), // (z + 1.0f) * n2
    };

    if P::POLICY.check_overflow {
        let in_range = x0.abs().cmp_lt(Vf::splat(max_x)) & x0.is_finite();

        if crate::likely(in_range.all()) {
            return z;
        }

        let underflow_value = const { if MODE == EXP_MODE_EXPM1 { Vf::NEG_ONE } else { Vf::ZERO } };

        r = x0.select_negative(underflow_value, Vf::INFINITY);
        z = in_range.select(z, r);
        z = x0.is_nan().select(x0, z);
    }

    z
}
