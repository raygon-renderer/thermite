use super::*;

impl<R> MathInternal<f32> for R
where
    R: FloatRegister<Element = f32>,
{
    fn sincos<P: Policy>(xx: Vf<Self>) -> (Vf<Self>, Vf<Self>) {
        use num_traits::FloatConst;

        let dp1f = const { Vector::splat_const(0.78515625 * 2.0) };
        let dp2f = const { Vector::splat_const(2.4187564849853515625E-4 * 2.0) };
        let dp3f = const { Vector::splat_const(3.77489497744594108E-8 * 2.0) };

        let xa = xx.abs();

        let frac_2_pi = Vf::<Self>::FRAC_2_PI();

        let y = (xa * frac_2_pi).round();
        let q: Vu<Self> = Vs::<Self>::fast_from(y).into_bits();

        // Reduce by extended precision modular arithmetic
        // x = ((xa - y * DP1F) - y * DP2F) - y * DP3F;
        let x = y.nmul_adde(dp3f, y.nmul_adde(dp2f, y.nmul_adde(dp1f, xa)));

        // Taylor expansion of sin and cos, valid for -pi/4 <= x <= pi/4
        let x2 = x * x;

        #[rustfmt::skip]
        let mut s = fast_polynomial::poly_array(x2, &[
            const { Vector::splat_const(-1.6666654611E-1) },
            const { Vector::splat_const(8.3321608736E-3) },
            const { Vector::splat_const(-1.9515295891E-4) },
        ])
        .mul_adde(x2 * x, x);

        let one_half = const { Vf::<Self>::splat_const(0.5) };

        #[rustfmt::skip]
        let mut c = fast_polynomial::poly_array(x2, &[
            const { Vector::splat_const(4.166664568298827E-2) },
            const { Vector::splat_const(-1.388731625493765E-3) },
            const { Vector::splat_const(2.443315711809948E-5) },
        ])
        .mul_adde(x2 * x2, one_half.nmul_adde(x2, Vf::<Self>::ONE));

        let swap = (q & Vu::<Self>::ONE).cmp_ne(Vu::<Self>::ZERO);

        let sin1 = swap.select(s, c);
        let cos1 = swap.select(c, s);

        let signsin = Vf::<Self>::from_bits(q.shli::<30>()) ^ xx;
        let signcos = Vf::<Self>::from_bits(((q + Vu::<Self>::ONE) & Vu::<Self>::TWO).shli::<30>());

        (sin1.combine_sign(signsin), (cos1 ^ signcos))
    }
}
