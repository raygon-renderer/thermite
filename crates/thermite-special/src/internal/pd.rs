use thermite::{
    mask::Mask,
    math::{
        MathWithPolicy,
        policy::{
            PrecisionPolicy,
            policies::{AveragePrecision, CmpLessPrecision, ExtraPrecision, MediumPrecision, ReferencePrecision},
        },
    },
};

use super::*;

impl<R> SpecialMathInternal<f64> for R
where
    R: FloatRegister<Element = f64>,
{
    fn tgamma<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        todo!()
    }

    fn lgamma_r<P: Policy>(x: Vf<Self>) -> (Vf<Self>, Vf<Self>) {
        todo!()
    }

    #[inline(always)]
    fn beta<P: Policy>(a: Vf<Self>, b: Vf<Self>) -> Vf<Self> {
        let is_valid = a.cmp_gt(Vf::ZERO) & b.cmp_gt(Vf::ZERO);

        if const { P::POLICY.check_overflow && !P::POLICY.avoid_branching } && is_valid.none() {
            return Vf::NAN;
        }

        let c = a + b;

        // if a < b then swap
        let (a, b) = (a.max(b), a.min(b));

        let mut result = a.poly_rational_p::<P, _, _>(&LANCZOS_P_EXPG_SCALED, &LANCZOS_Q)
            * (b.poly_rational_p::<P, _, _>(&LANCZOS_P_EXPG_SCALED, &LANCZOS_Q)
                / c.poly_rational_p::<P, _, _>(&LANCZOS_P_EXPG_SCALED, &LANCZOS_Q));

        let gh = Vf::splat(LANCZOS_G - 0.5);

        let agh = a + gh;
        let bgh = b + gh;
        let cgh = c + gh;

        let agh_d_cgh = agh / cgh;
        let bgh_d_cgh = bgh / cgh;
        let agh_p_bgh = agh * bgh;
        let cgh_p_cgh = cgh * cgh;

        let base = cgh
            .cmp_gt(Vf::splat(1e10))
            .select(agh_d_cgh * bgh_d_cgh, agh_p_bgh / cgh_p_cgh);

        // encourage instruction-level parallelism
        result *= agh_d_cgh.powf_p::<P>(a - Vf::HALF - b) * base.powf_p::<P>(b) * (Vf::SQRT_E / bgh.sqrt());

        if P::POLICY.check_overflow {
            result = is_valid.select(result, Vf::NAN);
        }

        result
    }
}

const LANCZOS_G: f64 = 6.024680040776729583740234375;

const LANCZOS_P: [f64; 13] = [
    23531376880.41075968857200767445163675473,
    42919803642.64909876895789904700198885093,
    35711959237.35566804944018545154716670596,
    17921034426.03720969991975575445893111267,
    6039542586.352028005064291644307297921070,
    1439720407.311721673663223072794912393972,
    248874557.8620541565114603864132294232163,
    31426415.58540019438061423162831820536287,
    2876370.628935372441225409051620849613599,
    186056.2653952234950402949897160456992822,
    8071.672002365816210638002902272250613822,
    210.8242777515793458725097339207133627117,
    2.506628274631000270164908177133837338626,
];

const LANCZOS_Q: [f64; 13] = [
    0.0,
    39916800.0,
    120543840.0,
    150917976.0,
    105258076.0,
    45995730.0,
    13339535.0,
    2637558.0,
    357423.0,
    32670.0,
    1925.0,
    66.0,
    1.0,
];

const LANCZOS_P_EXPG_SCALED: [f64; 13] = [
    56906521.91347156388090791033559122686859,
    103794043.1163445451906271053616070238554,
    86363131.28813859145546927288977868422342,
    43338889.32467613834773723740590533316085,
    14605578.08768506808414169982791359218571,
    3481712.15498064590882071018964774556468,
    601859.6171681098786670226533699352302507,
    75999.29304014542649875303443598909137092,
    6955.999602515376140356310115515198987526,
    449.9445569063168119446858607650988409623,
    19.51992788247617482847860966235652136208,
    0.5098416655656676188125178644804694509993,
    0.006061842346248906525783753964555936883222,
];
