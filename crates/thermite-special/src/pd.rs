use super::*;

use core::f64::consts::{FRAC_PI_2, FRAC_PI_4, LN_10, LN_2, LOG10_2, LOG10_E, LOG2_E, PI, SQRT_2};

const EULERS_CONSTANT: f64 = 5.772156649015328606065120900824024310e-01;
const LN_PI: f64 = 1.1447298858494001741434273513530587116472948129153115715136230714;
const SQRT_E: f64 = 1.6487212707001281468486507878141635716537761007101480115750793116;

impl<S: Simd> SimdVectorizedSpecialFunctionsInternal<S> for f64
where
    <S as Simd>::Vf64: SimdFloatVector<S, Element = f64>,
{
    #[inline(always)]
    fn tgamma<P: Policy>(mut z: Self::Vf) -> Self::Vf {
        let zero = Vf64::<S>::zero();
        let one = Vf64::<S>::one();
        let half = Vf64::<S>::splat(0.5);
        let quarter = Vf64::<S>::splat(0.25);
        let pi = Vf64::<S>::splat(PI);

        let orig_z = z;

        let is_neg = z.is_negative();
        let mut reflected = Mask::falsey();

        let mut res = one;

        'goto_positive: while is_neg.any() {
            reflected = z.le(Vf64::<S>::splat(-20.0));

            let mut refl_res = unsafe { Vf64::<S>::undefined() };

            // sine is expensive, so branch for it.
            if P::POLICY.avoid_precision_branches() || thermite_unlikely!(reflected.any()) {
                refl_res = <Self as SimdVectorizedMathInternal<S>>::sin_pix::<P>(z);

                // If not branching, all negative values are reflected
                if P::POLICY.avoid_precision_branches() {
                    reflected = is_neg;

                    res = reflected.select(refl_res, res);
                    z = z.conditional_neg(reflected);

                    break 'goto_positive;
                }

                // NOTE: I chose not to use a bitmask here, because some bitmasks can be
                // one extra instruction than the raw call to `all` again, and since z <= -20 is so rare,
                // that extra instruction is not worth it.
                if reflected.all() {
                    res = refl_res;
                    z = -z;

                    break 'goto_positive;
                }
            }

            let mut mod_z = z;
            let mut is_neg = is_neg;

            // recursively apply Γ(z+1)/z
            while is_neg.any() {
                res = is_neg.select(res / mod_z, res);
                mod_z = mod_z.conditional_add(one, is_neg);
                is_neg = mod_z.is_negative();
            }

            z = reflected.select(-z, mod_z);
            res = reflected.select(refl_res, res);

            break 'goto_positive;
        }

        // label
        //positive:

        // Integers

        let mut z_int = Mask::falsey();
        let mut fact_res = one;

        if P::POLICY.precision > PrecisionPolicy::Worst {
            let zf = z.floor();
            z_int = zf.eq(z);

            let bitmask = z_int.bitmask();

            if thermite_unlikely!(bitmask.any()) {
                let mut j = one;
                let mut k = j.lt(zf);

                while k.any() {
                    fact_res = k.select(fact_res * j, fact_res);
                    j += one;
                    k = j.lt(zf);
                }

                // Γ(-int) = NaN for poles
                fact_res = is_neg.select(Vf64::<S>::nan(), fact_res);
                // approaching zero from either side results in +/- infinity
                fact_res = orig_z.eq(zero).select(Vf64::<S>::infinity().copysign(orig_z), fact_res);

                if bitmask.all() {
                    return fact_res;
                }
            }
        }

        // Full

        let gh = Vf64::<S>::splat(LANCZOS_G - 0.5);

        let lanczos_sum = z.poly_rational_p::<P>(LANCZOS_P, LANCZOS_Q);

        let zgh = z + gh;
        let lzgh = zgh.ln_p::<P>();

        // (z * lzfg) > ln(f64::MAX)
        let very_large = (z * lzgh).gt(Vf64::<S>::splat(
            709.78271289338399672769243071670056097572649130589734950577761613,
        ));

        // only compute powf once
        let h = zgh.powf_p::<P>(very_large.select(z.mul_sube(half, quarter), z - half));

        // save a couple cycles by avoiding this division, but worst-case precision is slightly worse
        let denom = if P::POLICY.precision >= PrecisionPolicy::Best {
            lanczos_sum / zgh.exp_p::<P>()
        } else {
            lanczos_sum * (-zgh).exp_p::<P>()
        };

        let normal_res = very_large.select(h * h, h) * denom;

        // Tiny
        if P::POLICY.precision >= PrecisionPolicy::Best {
            let is_tiny = z.lt(Vf64::<S>::splat_as(
                <Self as SimdVectorizedMathInternal<S>>::__SQRT_EPSILON,
            ));
            let tiny_res = one / z - Vf64::<S>::splat(EULERS_CONSTANT);

            res *= is_tiny.select(tiny_res, normal_res);
        } else {
            res *= normal_res;
        }

        reflected.select(-pi / res, z_int.select(fact_res, res))
    }

    #[inline(always)]
    fn lgamma<P: Policy>(mut z: Self::Vf) -> Self::Vf {
        let one = Vf64::<S>::one();
        let zero = Vf64::<S>::zero();

        let reflect = z.lt(zero);

        let mut t = one;

        // NOTE: instead of computing ln(t) here, it's deferred to below where instead of
        // ln_pi - ln(gamma(x)) - ln(t), we compute ln_pi - ln(gamma(x) * t), sort of.
        // It's actually on the lanczos sum rather than gamma, but same difference.
        if P::POLICY.avoid_branching || reflect.any() {
            t = reflect.select(<Self as SimdVectorizedMathInternal<S>>::sin_pix::<P>(z).abs(), one);
            z = z.conditional_neg(reflect);
        }

        let gh = Vf64::<S>::splat(LANCZOS_G - 0.5);

        let mut lanczos_sum = z.poly_rational_p::<P>(LANCZOS_P_EXPG_SCALED, LANCZOS_Q);

        // Full A
        let mut a = (z + gh).ln_p::<P>() - one;

        // Tiny
        if P::POLICY.precision >= PrecisionPolicy::Best {
            let is_not_tiny = z.ge(Vf64::<S>::splat_as(
                <Self as SimdVectorizedMathInternal<S>>::__SQRT_EPSILON,
            ));
            let tiny_res = one / z - Vf64::<S>::splat(EULERS_CONSTANT);

            // shove the tiny result into the log down below
            lanczos_sum = is_not_tiny.select(lanczos_sum, tiny_res);
            // force multiplier to zero for tiny case, allowing the modified
            // lanczos sum and ln(t) to be combined for cheap
            a &= is_not_tiny.value();
        }

        // Full

        let b = z - Vf64::<S>::splat(0.5);
        let c = (lanczos_sum * t).ln_p::<P>();

        let mut res = a.mul_adde(b, c);

        let ln_pi = Vf64::<S>::splat(LN_PI);

        res = reflect.select(ln_pi - res, res);

        res
    }

    #[inline(always)]
    fn digamma<P: Policy>(mut x: Self::Vf) -> Self::Vf {
        let zero = Vf64::<S>::zero();
        let one = Vf64::<S>::one();
        let pi = Vf64::<S>::splat(PI);

        let mut result = zero;

        let reflect = x.le(Vf64::<S>::neg_one());

        if reflect.any() {
            x = reflect.select(one - x, x);

            let mut rem = x - x.floor();

            rem = rem.conditional_sub(one, rem.gt(Vf64::<S>::splat(0.5)));

            let (s, c) = (rem * pi).sin_cos_p::<P>();
            let refl_res = pi * c / s;

            result = reflect.select(refl_res, result);
        }

        let lim = Vf64::<S>::splat(
            0.5 * (10 + ((<Self as SimdVectorizedMathInternal<S>>::__DIGITS as i64 - 50) * 240) / 950) as f64,
        );

        // Rescale to use asymptotic expansion
        let mut is_small = x.lt(lim);
        while is_small.any() {
            result = result.conditional_sub(one / x, is_small);
            x = x.conditional_add(one, is_small);
            is_small = x.lt(lim);
        }

        x -= one;

        let z = one / (x * x);
        let a = x.ln_p::<P>() + (one / (x + x));

        let y = z.poly_p::<P>(&[
            0.083333333333333333333333333333333333333333333333333,
            -0.0083333333333333333333333333333333333333333333333333,
            0.003968253968253968253968253968253968253968253968254,
            -0.0041666666666666666666666666666666666666666666666667,
            0.0075757575757575757575757575757575757575757575757576,
            -0.021092796092796092796092796092796092796092796092796,
            0.083333333333333333333333333333333333333333333333333,
            -0.44325980392156862745098039215686274509803921568627,
        ]);

        result += z.nmul_adde(y, a);

        result
    }

    #[inline(always)]
    fn beta<P: Policy>(a: Self::Vf, b: Self::Vf) -> Self::Vf {
        let zero = Vf64::<S>::zero();

        let is_valid = a.gt(zero) & b.gt(zero);

        if P::POLICY.check_overflow && !P::POLICY.avoid_branching {
            if is_valid.none() {
                return Vf64::<S>::nan();
            }
        }

        let c = a + b;

        // if a < b then swap
        let (a, b) = (a.max(b), a.min(b));

        let mut result = a.poly_rational_p::<P>(LANCZOS_P_EXPG_SCALED, LANCZOS_Q)
            * (b.poly_rational_p::<P>(LANCZOS_P_EXPG_SCALED, LANCZOS_Q)
                / c.poly_rational_p::<P>(LANCZOS_P_EXPG_SCALED, LANCZOS_Q));

        let gh = Vf64::<S>::splat(LANCZOS_G - 0.5);

        let agh = a + gh;
        let bgh = b + gh;
        let cgh = c + gh;

        let agh_d_cgh = agh / cgh;
        let bgh_d_cgh = bgh / cgh;
        let agh_p_bgh = agh * bgh;
        let cgh_p_cgh = cgh * cgh;

        let base = cgh
            .gt(Vf64::<S>::splat(1e10))
            .select(agh_d_cgh * bgh_d_cgh, agh_p_bgh / cgh_p_cgh);

        // encourage instruction-level parallelism
        result *= agh_d_cgh.powf_p::<P>(a - Vf64::<S>::splat(0.5) - b)
            * base.powf_p::<P>(b)
            * (Vf64::<S>::splat(SQRT_E) / bgh.sqrt());

        if P::POLICY.check_overflow {
            result = is_valid.select(result, Vf64::<S>::nan());
        }

        result
    }
}

const LANCZOS_G: f64 = 6.024680040776729583740234375;

const LANCZOS_P: &[f64] = &[
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

const LANCZOS_Q: &[f64] = &[
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

const LANCZOS_P_EXPG_SCALED: &[f64] = &[
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
