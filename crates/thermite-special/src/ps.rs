use super::*;

use core::f32::consts::{
    FRAC_1_PI, FRAC_2_PI, FRAC_PI_2, FRAC_PI_4, LN_10, LN_2, LOG10_2, LOG10_E, LOG2_E, PI, SQRT_2,
};

const EULERS_CONSTANT: f32 = 5.772156649015328606065120900824024310e-01;
const LN_PI: f32 = 1.1447298858494001741434273513530587116472948129153115715136230714;
const SQRT_E: f32 = 1.6487212707001281468486507878141635716537761007101480115750793116;

impl<S: Simd> SimdVectorizedSpecialFunctionsInternal<S> for f32
where
    <S as Simd>::Vf32: SimdFloatVector<S, Element = f32>,
{
    #[inline(always)]
    fn tgamma<P: Policy>(mut z: Self::Vf) -> Self::Vf {
        let zero = Vf32::<S>::zero();
        let one = Vf32::<S>::one();
        let half = Vf32::<S>::splat(0.5);
        let quarter = Vf32::<S>::splat(0.25);
        let pi = Vf32::<S>::splat(PI);

        let orig_z = z;

        let is_neg = z.is_negative();
        let mut reflected = Mask::falsey();

        let mut res = one;

        'goto_positive: while is_neg.any() {
            reflected = z.le(Vf32::<S>::splat(-20.0));

            let mut refl_res = unsafe { Vf32::<S>::undefined() };

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
                fact_res = is_neg.select(Vf32::<S>::nan(), fact_res);
                // approaching zero from either side results in +/- infinity
                fact_res = orig_z.eq(zero).select(Vf32::<S>::infinity().copysign(orig_z), fact_res);

                if bitmask.all() {
                    return fact_res;
                }
            }
        }

        // Full

        let gh = Vf32::<S>::splat(LANCZOS_G - 0.5);

        let lanczos_sum = z.poly_rational_p::<P>(LANCZOS_P, LANCZOS_Q);

        let zgh = z + gh;
        let lzgh = zgh.ln_p::<P>();

        // (z * lzfg) > ln(f32::MAX)
        let very_large = (z * lzgh).gt(Vf32::<S>::splat(
            88.722839053130621324601674778549183073943430402325230485234240247,
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
            let is_tiny = z.lt(Vf32::<S>::splat(
                <Self as SimdVectorizedMathInternal<S>>::__SQRT_EPSILON,
            ));
            let tiny_res = z.reciprocal_p::<P>() - Vf32::<S>::splat(EULERS_CONSTANT);
            res *= is_tiny.select(tiny_res, normal_res);
        } else {
            res *= normal_res;
        }

        reflected.select(-pi / res, z_int.select(fact_res, res))
    }

    #[inline(always)]
    fn lgamma<P: Policy>(mut z: Self::Vf) -> Self::Vf {
        let one = Vf32::<S>::one();
        let zero = Vf32::<S>::zero();

        let reflect = z.lt(zero);

        let mut t = one;

        if P::POLICY.avoid_branching || reflect.any() {
            t = reflect.select(<Self as SimdVectorizedMathInternal<S>>::sin_pix::<P>(z).abs(), one);
            z = z.conditional_neg(reflect);
        }

        let gh = Vf32::<S>::splat(LANCZOS_G - 0.5);

        let mut lanczos_sum = z.poly_rational_p::<P>(LANCZOS_P_EXPG_SCALED, LANCZOS_Q);

        // Full A
        let mut a = (z + gh).ln_p::<P>() - one;

        // Tiny
        if P::POLICY.precision >= PrecisionPolicy::Best {
            let is_not_tiny = z.ge(Vf32::<S>::splat_as(
                <Self as SimdVectorizedMathInternal<S>>::__SQRT_EPSILON,
            ));
            let tiny_res = z.reciprocal_p::<P>() - Vf32::<S>::splat(EULERS_CONSTANT);

            // shove the tiny result into the log down below
            lanczos_sum = is_not_tiny.select(lanczos_sum, tiny_res);
            // force multiplier to zero for tiny case, allowing the modified
            // lanczos sum and ln(t) to be combined for cheap
            a &= is_not_tiny.value();
        }

        // Full

        let b = z - Vf32::<S>::splat(0.5);
        let c = (lanczos_sum * t).ln_p::<P>();

        let mut res = a.mul_adde(b, c);

        let ln_pi = Vf32::<S>::splat(LN_PI);

        res = reflect.select(ln_pi - res, res);

        res
    }

    #[inline(always)]
    fn digamma<P: Policy>(mut x: Self::Vf) -> Self::Vf {
        let zero = Vf32::<S>::zero();
        let one = Vf32::<S>::one();
        let half = Vf32::<S>::splat(0.5);
        let pi = Vf32::<S>::splat(PI);

        let mut result = zero;

        let reflect = x.le(Vf32::<S>::neg_one());

        if reflect.any() {
            x = reflect.select(one - x, x);

            let mut rem = x - x.floor();

            rem = rem.conditional_sub(one, rem.gt(half));

            let (s, c) = (rem * pi).sin_cos_p::<P>();
            let refl_res = pi * c / s;

            result = reflect.select(refl_res, result);
        }

        let lim = Vf32::<S>::splat(
            0.5 * (10 + ((<Self as SimdVectorizedMathInternal<S>>::__DIGITS as i64 - 50) * 240) / 950) as f32,
        );

        // Rescale to use asymptotic expansion
        let mut is_small = x.lt(lim);
        while is_small.any() {
            result = result.conditional_sub(x.reciprocal_p::<P>(), is_small);
            x = x.conditional_add(one, is_small);
            is_small = x.lt(lim);
        }

        x -= one;

        let inv_x = x.reciprocal_p::<P>();

        let z = inv_x * inv_x;
        let a = x.ln_p::<P>() + (inv_x * half);

        let y = z.poly_p::<P>(&[
            0.083333333333333333333333333333333333333333333333333,
            -0.0083333333333333333333333333333333333333333333333333,
            0.003968253968253968253968253968253968253968253968254,
        ]);

        result += z.nmul_adde(y, a);

        result
    }

    #[inline(always)]
    fn beta<P: Policy>(a: Self::Vf, b: Self::Vf) -> Self::Vf {
        let zero = Vf32::<S>::zero();

        let is_valid = a.gt(zero) & b.gt(zero);

        if P::POLICY.check_overflow && !P::POLICY.avoid_branching {
            if is_valid.none() {
                return Vf32::<S>::nan();
            }
        }

        let c = a + b;

        // if a < b then swap
        let (a, b) = (a.max(b), a.min(b));

        let mut result = a.poly_rational_p::<P>(LANCZOS_P_EXPG_SCALED, LANCZOS_Q)
            * (b.poly_rational_p::<P>(LANCZOS_P_EXPG_SCALED, LANCZOS_Q)
                / c.poly_rational_p::<P>(LANCZOS_P_EXPG_SCALED, LANCZOS_Q));

        let gh = Vf32::<S>::splat(LANCZOS_G - 0.5);

        let agh = a + gh;
        let bgh = b + gh;
        let cgh = c + gh;

        let agh_d_cgh = agh / cgh;
        let bgh_d_cgh = bgh / cgh;
        let agh_p_bgh = agh * bgh;
        let cgh_p_cgh = cgh * cgh;

        let base = cgh
            .gt(Vf32::<S>::splat(1e10))
            .select(agh_d_cgh * bgh_d_cgh, agh_p_bgh / cgh_p_cgh);

        let denom = if P::POLICY.precision > PrecisionPolicy::Average {
            Vf32::<S>::splat(SQRT_E) / bgh.sqrt()
        } else {
            // bump up the precision a little to improve beta function accuracy
            Vf32::<S>::splat(SQRT_E) * bgh.invsqrt_p::<policies::ExtraPrecision<P>>()
        };

        result *= agh_d_cgh.powf_p::<P>(a - Vf32::<S>::splat(0.5) - b) * (base.powf_p::<P>(b) * denom);

        if P::POLICY.check_overflow {
            result = is_valid.select(result, Vf32::<S>::nan());
        }

        result
    }
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
