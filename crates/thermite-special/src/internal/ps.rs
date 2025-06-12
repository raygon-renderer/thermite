use thermite::math::{
    MathWithPolicy,
    policy::{PrecisionPolicy, policies::ExtraPrecision},
};

use super::*;

impl<R> SpecialMathInternal<f32> for R
where
    R: FloatRegister<Element = f32>,
{
    #[inline(always)]
    fn tgamma<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        let is_negative = x.is_negative();

        if P::POLICY.avoid_branching || is_negative.any() {
            // TODO: Negatives

            let x = x.abs();
        }

        tgamma0::<Self, P>(x)
    }
}

#[inline(always)]
fn tgamma0<R: FloatRegister<Element = f32>, P: Policy>(x: Vf<R>) -> Vf<R> {
    if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
        /**
         * "An accurate approximation formula for gamma function"
         * https://journalofinequalitiesandapplications.springeropen.com/articles/10.1186/s13660-018-1646-6
         *
         * This has been reformulated to use a single exponential function call and single logarithm call,
         * by moving all the terms into the exponent and using a Padé approximation for the ln/exp generated
         * by expanding sinh.
         */
        let xr = x.reciprocal_p::<P>();

        // (3x+1)/2 * ln(x) - x
        let a = x
            .mul_adde(Vf::splat(1.5), Vf::HALF)
            .mul_sube(x.ln_p::<ExtraPrecision<P>>(), x);

        // (1-x)/2 * ln(2) + 1/2
        let b = x.nmul_adde(Vf::HALF, Vf::HALF).mul_adde(Vf::LN_2, Vf::HALF);

        // ln(pi)/2 + x/2 * ln(1 - e^(-2/x))
        let c = {
            let xr2 = xr + xr; // 2/x

            // ln(1 - e^(-x))/2 Padé approximation generated around 0.5 by WolframAlpha
            let hln1m_en2x = xr2.poly_p::<P, 3>(&[-1.01847808 / 2.0, -7.1853735 / 2.0, 1.31300828 / 2.0])
                / xr2.poly_p::<P, 5>(&[0.232837568, 4.2072807, 7.9589356, 1.62386099, 1.0]);

            hln1m_en2x.min(Vf::ZERO).mul_adde(x, Vf::FRAC_LN_PI_2)
        };

        // 7/324 * 1/(x^3(35x^2 + 33)) + c
        let d = {
            let c0 = Vf::splat(7.0 / 324.0);
            let c1 = Vf::splat(35.0);
            let c2 = Vf::splat(33.0);

            let x2 = x * x;
            let d0 = x2.mul_adde(c1, c2) * (x * x2);

            d0.reciprocal_p::<P>().mul_adde(c0, c)
        };

        // e^(a + b + d) / x
        return (a + b + d).exp_p::<P>() * xr;
    }

    unimplemented!()
}

const LANCZOS_G: f32 = 1.428456135094165802001953125;

const LANCZOS_P: &[f32] = &[
    58.52061591769095910314047740215847630266,
    182.5248962595894264831189414768236280862,
    211.0971093028510041839168287718170827259,
    112.2526547883668146736465390902227161763,
    27.5192015197455403062503721613097825345,
    2.50662858515256974113978724717473206342,
];

const LANCZOS_Q: &[f32] = &[0.0, 24.0, 50.0, 35.0, 10.0, 1.0];

const LANCZOS_P_EXPG_SCALED: &[f32] = &[
    14.0261432874996476619570577285003839357,
    43.74732405540314316089531289293124360129,
    50.59547402616588964511581430025589038612,
    26.90456680562548195593733429204228910299,
    6.595765571169314946316366571954421695196,
    0.6007854010515290065101128585795542383721,
];
