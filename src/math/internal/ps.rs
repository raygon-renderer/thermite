use crate::math::{consts::FloatConsts as _, policy::policies::ExtraPrecision};
use core::f32::consts::{FRAC_1_PI, LN_10, LOG2_E, SQRT_2};

use super::*;

impl<R> MathInternal<f32> for R
where
    R: FloatRegister<Element = f32>,
{
    #[inline(always)]
    fn sincos<P: Policy>(xx: Vf<Self>) -> (Vf<Self>, Vf<Self>) {
        let xa = xx.abs();

        let frac_2_pi = Vf::<Self>::FRAC_2_PI;

        let y = (xa * frac_2_pi).round();
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
        let mut s = x2.poly_p::<P, 3>(&[
            -1.6666654611E-1,
            8.3321608736E-3,
            -1.9515295891E-4,
        ])
        .mul_adde(x2 * x, x);

        let one_half = const { Vf::<Self>::splat_const(0.5) };

        #[rustfmt::skip]
        let mut c = x2.poly_p::<P, 3>(&[
            4.166664568298827E-2,
            -1.388731625493765E-3,
            2.443315711809948E-5,
        ])
        .mul_adde(x2 * x2, one_half.nmul_adde(x2, Vf::<Self>::ONE));

        let swap = (q & Vu::<Self>::ONE).cmp_ne(Vu::<Self>::ZERO);

        let sin1 = swap.select(c, s);
        let cos1 = swap.select(s, c);

        let signsin = Vf::<Self>::from_bits(q.shli::<30>()) ^ xx;
        let signcos = Vf::<Self>::from_bits(((q + Vu::<Self>::ONE) & Vu::<Self>::TWO).shli::<30>());

        (sin1.combine_sign(signsin), (cos1 ^ signcos))
    }

    #[inline(always)]
    fn sinh<P: Policy>(x0: Vf<Self>) -> Vf<Self> {
        let x = x0.abs();

        let x_small = x.cmp_lt(Vf::<Self>::ONE);

        let mut y1 = Vf::<Self>::EMPTY;
        let mut y2 = Vf::<Self>::EMPTY;

        // if not all are small, use exponential functions
        if P::POLICY.avoid_branching || !x_small.all() {
            y2 = Self::exph::<P>(x);
            y2 -= Vf::<Self>::splat(0.25) / y2;

            if const { P::POLICY.avoid_precision_branches() } {
                return y2.combine_sign(x0);
            }
        }

        // if all are small, use a polynomial approximation
        if P::POLICY.avoid_branching || x_small.any() {
            let x2 = x * x;

            y1 = x2
                .poly_p::<P, 3>(&[1.66667160211E-1, 8.33028376239E-3, 2.03721912945E-4])
                .mul_adde(x2 * x, x);
        }

        x_small.select(y1, y2).combine_sign(x0)
    }

    #[inline(always)]
    fn cosh<P: Policy>(x0: Vf<Self>) -> Vf<Self> {
        let y = Self::exph::<P>(x0.abs());
        y + Vf::<Self>::splat(0.25) / y
    }

    #[inline(always)]
    #[rustfmt::skip]
    fn tanh<P: Policy>(x0: Vf<Self>) -> Vf<Self> {
        let one = Vf::<Self>::ONE;

        let x = x0.abs();
        let x_small = x.cmp_lt(Vf::<Self>::splat(0.625));

        let mut y1 = Vf::<Self>::EMPTY;
        let mut y2 = Vf::<Self>::EMPTY;

        // if not all are small
        if P::POLICY.avoid_branching || !x_small.all() {
            y2 = Self::exp::<P>(x + x);
            // originally (1 - 2/(y2 + 1)), but doing it this way avoids
            // loading 2.0 and encourages slight instruction-level parallelism
            y2 = (y2 - one) / (y2 + one);

            if P::POLICY.check_overflow {
                y2 = x.cmp_gt(Vf::<Self>::splat(44.4)).select(one, y2);
            }

            if P::POLICY.avoid_precision_branches() {
                return y2.combine_sign(x0);
            }
        }

        // if any are small
        if P::POLICY.avoid_branching || x_small.any() {
            let x2 = x * x;

            y1 = x2.poly_p::<P, 5>(&[
                -3.33332819422E-1,
                1.33314422036E-1,
                -5.37397155531E-2,
                2.06390887954E-2,
                -5.70498872745E-3,
            ]).mul_adde(x2 * x, x);
        }

        x_small.select(y1, y2).combine_sign(x0)
    }

    #[inline(always)]
    fn asin<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        asin_f_internal::<P, Self, false>(x)
    }

    #[inline(always)]
    fn acos<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        asin_f_internal::<P, Self, true>(x)
    }

    fn atan<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn atan2<P: Policy>(y: Vf<Self>, x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn asinh<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn acosh<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn atanh<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
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
        if P::POLICY.precision == PrecisionPolicy::Worst {
            return (x0.log2_p::<P>() * y).exp2_p::<P>();
        }

        // define constants
        let ln2f_hi = Vf::<R>::splat(0.693359375); // log(2), split in two for extended precision
        let ln2f_lo = Vf::<R>::splat(-2.12194440e-4);
        let log2e = Vf::<R>::LOG2_E;
        let ln2 = Vf::<R>::LN_2;

        let zero = Vf::<R>::ZERO;
        let one = Vf::<R>::ONE;
        let half = Vf::<R>::HALF;

        let x1 = x0.abs();

        let mut x = fraction2::<R>(x1);

        let blend = x.cmp_gt(Vf::<R>::splat(SQRT_2 * 0.5));

        // reduce range of x = +/- sqrt(2)/2
        x += blend.andnot(x); // !blend.value() & x;
        x -= one;

        // Taylor expansion, high precision
        let x2 = x * x;

        // logarithm expansion
        let mut lg1 = x.poly_p::<P, 9>(&[
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
            .poly_p::<P, 6>(&[1.0 / 2.0, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0, 1.0 / 720.0, 1.0 / 5040.0])
            .mul_adde(x * x, x + one);

        // contributions to exponent
        let ee = e1 + e2 + e3;
        let ei = Vs::<R>::from(ee);

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

        if crate::unlikely((overflow | underflow).any()) {
            z = underflow.select(zero, z);
            z = overflow.select(Vf::INFINITY, z);
        }

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

    fn cbrt<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
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
        ln_f_internal::<P, Self, false>(x) * Vf::<Self>::LOG2_E
    }

    #[inline(always)]
    fn log10<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        ln_f_internal::<P, Self, false>(x) * Vf::<Self>::LOG10_2
    }

    #[inline(always)]
    fn erf<P: Policy>(x0: Vf<Self>) -> Vf<Self> {
        if const { P::POLICY.precision.eq(PrecisionPolicy::Reference) } {
            // Use erf(x) = 1 - erfc(x)
            return Vf::ONE - Self::erfc::<P>(x0);
        }

        let mut x = x0.abs();

        if P::POLICY.check_overflow {
            x = Vf::ONE - (Vf::ONE - x); // crush denormals
        }

        let y = match P::POLICY.precision {
            // 5 * 10^-4 accuracy
            PrecisionPolicy::Worst => {
                let t = x.poly_p::<P, 5>(&[1.0, 0.278393, 0.230389, 0.000972, 0.078108]);
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
                let r = x.poly_p::<P, 7>(&[
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
            //     let y = t.poly_p::<P, 5>(&[0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]);
            //
            //     y.nmul_adde(e, Vf::ONE)
            // }

            // 1.2 * 10^-7 accuracy
            PrecisionPolicy::Best => {
                // 1 / (1 + 1/2|x|)
                let t = x.mul_adde(Vf::HALF, Vf::ONE).reciprocal_p::<P>();

                let r0 = t.poly_p::<P, 10>(&[
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

        y.combine_sign(x0)
    }

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

        let y = n / d * (-x2).exp2_p::<P>();

        // if x<0 then 2 - y, else y
        x0.select_negative(Vf::TWO - y, y)
    }

    fn erfinv<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!("erfinv")
    }
}

#[inline(always)]
fn asin_f_internal<P: Policy, R: MathInternal<f32>, const ACOS: bool>(x: Vf<R>) -> Vf<R> {
    let xa = x.abs();

    let is_big = xa.cmp_gt(Vf::<R>::HALF);

    // TODO: Branch to avoid sqrt?
    let x1 = Vf::<R>::HALF * (Vf::<R>::ONE - xa);
    let x3 = is_big.select(x1, xa * xa);
    let x4 = is_big.select(x1.sqrt(), xa);

    #[rustfmt::skip]
    let z = x3.poly_p::<P, 5>(&[
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
        let z2 = Vf::<R>::FRAC_PI_2 - z.combine_sign(x);

        is_big.select(z1, z2)
    } else {
        let z1 = Vf::<R>::FRAC_PI_2 - z1;

        is_big.select(z1, z).combine_sign(x)
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
        let i: Vs<R> = fi.fast_cast(); // unsafe { fi.to_int_fast() };

        // polynomial approximation of 2^f
        let cf = f.poly_p::<P, 4>(&[1.0, 0.695556856, 0.226173572, 0.0781455737]);

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
            EXP_MODE_EXP | EXP_MODE_POW2 | EXP_MODE_POW10 => {
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
            .poly_p::<P, 6>(&[1.0 / 2.0, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0, 1.0 / 720.0, 1.0 / 5040.0])
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
fn ln_f_internal<P: Policy, R: MathInternal<f32>, const P1: bool>(x0: Vf<R>) -> Vf<R> {
    if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
        // https://stackoverflow.com/a/39822314/2083075

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
    let one = Vf::ONE;

    let x1 = if P1 { x0 + one } else { x0 };

    let mut x = fraction2::<R>(x1);
    let mut e = exponent::<R>(x1);

    let blend = x.cmp_gt(Vf::splat(SQRT_2 * 0.5));

    //x = x.conditional_add(x, !blend);
    //e = e.conditional_add(Vs::<R>::ONE, blend);

    let fe: Vf<R> = e.cast();

    let xp1 = x - one;

    x = if P1 {
        // log(x+1). Avoid loss of precision when adding 1 and later subtracting 1 if exponent = 0
        e.cmp_eq(Vs::<R>::ZERO).select(x0, xp1)
    } else {
        // log(x). Expand around 1.0
        xp1
    };

    let x2 = x * x;
    let mut res = x.poly_p::<P, 10>(&[
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
    res += x2.nmul_adde(Vf::HALF, x);
    res = fe.mul_adde(ln2f_hi, res);

    if const { !P::POLICY.check_overflow } {
        return res;
    }

    let overflow = !x1.is_finite();
    let underflow = x1.cmp_lt(Vf::<R>::splat(1.17549435e-38));

    if crate::likely((overflow | underflow).none()) {
        return res;
    }

    res = underflow.select(Vf::NAN, res); // x1 < 0 gives NAN
    res = x1.is_zero_or_subnormal().select(Vf::NEG_INFINITY, res); // x1 == 0 gives -INF
    res = overflow.select(x1, res); // INF or NAN goes through
    res = (x1.is_infinite() & x1.is_negative()).select(Vf::NAN, res); // -INF gives NAN

    res
}
