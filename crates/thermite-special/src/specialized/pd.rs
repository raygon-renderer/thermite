use thermite::{
    generic::GenericMask as _,
    mask::Mask,
    math::{
        TranscendentalMathWithPolicy,
        policy::{
            PrecisionPolicy,
            policies::{AveragePrecision, CmpLessPrecision, ExtraPrecision, MediumPrecision, ReferencePrecision},
        },
        specialized::SpecializedTranscendentalMath,
    },
};

use super::*;

impl<V> SpecializedSpecialMath<f64> for V
where
    V: TranscendentalMathWithPolicy<Element = f64>,
    V: SpecializedTranscendentalMath<f64>,
{
    #[inline(always)]
    fn erf<P: Policy>(self) -> Self {
        let x = self;
        let x2 = x * x;
        let res = x * x2.poly_rational_p::<P, _, _>(
            &[
                5.55923013010394962768e4,
                7.00332514112805075473e3,
                2.23200534594684319226e3,
                9.00260197203842689217e1,
                9.60497373987051638749e0,
                0.0,
            ],
            &[
                4.92673942608635921086e4,
                2.26290000613890934246e4,
                4.59432382970980127987e3,
                5.21357949780152679795e2,
                3.35617141647503099647e1,
                1.00000000000000000000e0,
            ],
        );

        if P::POLICY.check_overflow {
            // x^2 highest point in the polynomial, use x2 to avoid needing absolute value
            // TODO: Find more exact value?
            x2.cmp_gt(V::splat(8.135455562428929)).select(x.signum(), res)
        } else {
            res
        }
    }

    #[inline(always)]
    fn erfinv<P: Policy>(self) -> Self {
        let y = self;
        let a = y.abs();

        let w = -a.nmul_adde(a, V::ONE).ln_p::<P>();

        // https://www.desmos.com/calculator/yduhxx1ukm values extracted via JS console
        let mut p0 = (w - V::splat(2.5)).poly_p::<P, _>(&[
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

        let w_big = w.cmp_ge(V::splat(5.0)); // at around |x| > 0.99662533231, so unlikely

        if P::POLICY.avoid_branching || thermite::unlikely(w_big.any()) {
            let mut p1 = (w.sqrt() - V::splat(3.0)).poly_p::<P, _>(&[
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

            if P::POLICY.check_overflow {
                p1 = a.cmp_eq(V::ONE).select(V::INFINITY, p1); // erfinv(x == 1) = inf
                p1 = a.cmp_gt(V::ONE).select(V::NAN, p1); // erfinv(x > 1) = NaN
            }

            p0 = w_big.select(p1, p0);
        }

        p0 * y
    }

    fn tgamma<P: Policy>(x: Self) -> Self {
        todo!()
    }

    fn lgamma_r<P: Policy>(x: Self) -> (Self, Self) {
        todo!()
    }

    #[inline(always)]
    fn beta<P: Policy>(a: Self, b: Self) -> Self {
        let is_valid = a.cmp_gt(Self::ZERO) & b.cmp_gt(Self::ZERO);

        if const { P::POLICY.check_overflow && !P::POLICY.avoid_branching } && is_valid.none() {
            return Self::NAN;
        }

        let c = a + b;

        // if a < b then swap
        let (a, b) = (a.max(b), a.min(b));

        let mut result = a.poly_rational_p::<P, _, _>(&LANCZOS_P_EXPG_SCALED, &LANCZOS_Q)
            * (b.poly_rational_p::<P, _, _>(&LANCZOS_P_EXPG_SCALED, &LANCZOS_Q)
                / c.poly_rational_p::<P, _, _>(&LANCZOS_P_EXPG_SCALED, &LANCZOS_Q));

        let gh = Self::splat(LANCZOS_G - 0.5);

        let agh = a + gh;
        let bgh = b + gh;
        let cgh = c + gh;

        let agh_d_cgh = agh / cgh;
        let bgh_d_cgh = bgh / cgh;
        let agh_p_bgh = agh * bgh;
        let cgh_p_cgh = cgh * cgh;

        let base = cgh
            .cmp_gt(Self::splat(1e10))
            .select(agh_d_cgh * bgh_d_cgh, agh_p_bgh / cgh_p_cgh);

        // encourage instruction-level parallelism
        result *= agh_d_cgh.powf_p::<P>(a - Self::HALF - b) * base.powf_p::<P>(b) * (Self::SQRT_E / bgh.sqrt());

        if P::POLICY.check_overflow {
            result = is_valid.select(result, Self::NAN);
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
