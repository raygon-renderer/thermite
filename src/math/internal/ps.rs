use crate::math::consts::FloatConsts as _;

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
        let mut s = Self::poly::<P, 3>(x2, &[
            -1.6666654611E-1,
            8.3321608736E-3,
            -1.9515295891E-4,
        ])
        .mul_adde(x2 * x, x);

        let one_half = const { Vf::<Self>::splat_const(0.5) };

        #[rustfmt::skip]
        let mut c = Self::poly::<P, 3>(x2, &[
            4.166664568298827E-2,
            -1.388731625493765E-3,
            2.443315711809948E-5,
        ])
        .mul_adde(x2 * x2, one_half.nmul_adde(x2, Vf::<Self>::ONE));

        let swap = (q & Vu::<Self>::ONE).cmp_ne(Vu::<Self>::ZERO);

        let sin1 = swap.select(s, c);
        let cos1 = swap.select(c, s);

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

            y1 = Self::poly::<P, 3>(x2, &[1.66667160211E-1, 8.33028376239E-3, 2.03721912945E-4]).mul_adde(x2 * x, x);
        }

        x_small.select(y2, y1).combine_sign(x0)
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
                y2 = x.cmp_gt(Vf::<Self>::splat(44.4)).select(y2, one);
            }

            if P::POLICY.avoid_precision_branches() {
                return y2.combine_sign(x0);
            }
        }

        // if any are small
        if P::POLICY.avoid_branching || x_small.any() {
            let x2 = x * x;

            y1 = Self::poly::<P, 5>(x2, &[
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
        asin_internal::<P, Self, false>(x)
    }

    #[inline(always)]
    fn acos<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        asin_internal::<P, Self, true>(x)
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

    fn exp<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn exph<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn exp2<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn exp10<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn exp_m1<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn powf<P: Policy>(x: Vf<Self>, e: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn cbrt<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn ln<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn ln1p<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn log2<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn log10<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn erf<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn erfc<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn erfinv<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }
}

#[inline(always)]
fn asin_internal<P: Policy, R: MathInternal<f32>, const ACOS: bool>(x: Vf<R>) -> Vf<R> {
    let xa = x.abs();

    let is_big = xa.cmp_gt(Vf::<R>::splat(0.5));

    // TODO: Branch to avoid sqrt?
    let x1 = Vf::<R>::splat(0.5) * (Vf::<R>::ONE - xa);
    let x3 = is_big.select(xa * xa, x1);
    let x4 = is_big.select(xa, x1.sqrt());

    #[rustfmt::skip]
    let z = R::poly::<P, 5>(x3, &[
        1.6666752422E-1,
        7.4953002686E-2,
        4.5470025998E-2,
        2.4181311049E-2,
        4.2163199048E-2,
    ])
    .mul_adde(x3 * x4, x4);

    let z1 = z + z;

    if ACOS {
        let z1 = x.select_negative(z1, Vf::<R>::PI - z1);
        let z2 = Vf::<R>::FRAC_PI_2 - z.combine_sign(x);

        is_big.select(z2, z1)
    } else {
        let z1 = Vf::<R>::FRAC_PI_2 - z1;

        is_big.select(z, z1).combine_sign(x)
    }
}
