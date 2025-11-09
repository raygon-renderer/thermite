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

impl<R> SpecialMathInternal<f32> for R
where
    R: FloatRegister<Element = f32>,
{
    #[inline(always)]
    fn tgamma<P: Policy>(mut z: Vf<Self>) -> Vf<Self> {
        if const { P::POLICY.precision.le(PrecisionPolicy::Average) } {
            // We have a good lgamma approximation, so use it for tgamma on lower precisions.
            let (lgamma, sign) = z.lgamma_r_p::<P>();

            // use min(P + 1, Average) precision here. We want decent precision,
            // but not more than average.
            return lgamma.exp_p::<ExtraPrecision<P>>() * sign;
        }

        let orig_z = z;

        let is_negative = z.is_negative();
        let mut reflected = Mask::<R>::FALSY;

        let mut res = Vf::ONE;

        #[allow(clippy::never_loop)]
        'goto_positive: while is_negative.any() {
            reflected = z.cmp_le(Vf::splat(-20.0));

            let mut refl_res = Vf::EMPTY;

            if P::POLICY.avoid_precision_branches() || thermite::unlikely(reflected.any()) {
                refl_res = z.sin_pix_p::<P>();

                // If not branching, all negative values are reflected
                if const { P::POLICY.avoid_precision_branches() } {
                    reflected = is_negative;

                    res = reflected.select(refl_res, res);
                    z = z.abs();

                    break 'goto_positive;
                }

                if reflected.all() {
                    res = refl_res;
                    z = -z;

                    break 'goto_positive;
                }
            }

            let mut mod_z = z;
            let mut is_neg = is_negative;

            while is_neg.any() {
                res = is_neg.select(res / mod_z, res);
                mod_z += Vf::ONE & is_neg.value();
                is_neg = mod_z.is_negative();
            }

            z = reflected.select(-z, mod_z);
            res = reflected.select(refl_res, res);

            break 'goto_positive;
        }

        // Integers

        let mut is_int = Mask::<R>::FALSY;
        let mut int_res = Vf::ONE;

        if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            let zf = z.floor();
            is_int = zf.cmp_eq(z);

            if thermite::unlikely(is_int.any()) {
                let mut j = Vf::ONE;
                let mut k = j.cmp_lt(zf);

                while k.any() {
                    int_res = k.select(int_res * j, int_res);
                    j += Vf::ONE;
                    k = j.cmp_lt(zf);
                }

                // Γ(-int) = NaN for poles
                int_res = is_negative.select(Vf::NAN, int_res);

                // approaching zero from either side results in +/- infinity
                int_res = orig_z
                    .cmp_eq(Vf::ZERO)
                    .select(is_negative.select(Vf::NEG_INFINITY, Vf::INFINITY), int_res);

                if thermite::unlikely(is_int.all()) {
                    return int_res; // can skip full gamma calculation if all inputs are integers
                }
            }
        }

        // Full

        let gh = Vf::splat(LANCZOS_G - 0.5);

        let lanczos_sum = z.poly_rational_p::<P, _, _>(&LANCZOS_P, &LANCZOS_Q);

        let zgh = z + gh;
        let lzgh = zgh.ln_p::<P>();

        // (z * lzfg) > ln(f32::MAX)
        let very_large = (z * lzgh).cmp_gt(Vf::splat(
            88.722839053130621324601674778549183073943430402325230485234240247,
        ));

        // only compute powf once
        let h = zgh.powf_p::<P>(very_large.select(z.mul_sube(Vf::HALF, Vf::splat(0.25)), z - Vf::HALF));

        // save a couple cycles by avoiding this division, but worst-case precision is slightly worse
        let denom = if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            lanczos_sum / zgh.exp_p::<P>()
        } else {
            lanczos_sum * (-zgh).exp_p::<P>()
        };

        let normal_res = very_large.select(h * h, h) * denom;

        // Tiny
        if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            let is_tiny = z.cmp_lt(Vf::SQRT_EPSILON);
            let tiny_res = z.reciprocal_p::<P>() - Vf::EGAMMA;
            res *= is_tiny.select(tiny_res, normal_res);
        } else {
            res *= normal_res;
        }

        reflected.select(-Vf::PI / res, is_int.select(int_res, res))
    }

    #[inline(always)]
    fn lgamma_r<P: Policy>(mut z: Vf<Self>) -> (Vf<Self>, Vf<Self>) {
        let mut signum = Vf::ONE;

        let reflect = z.is_negative();

        if const { P::POLICY.precision.le(PrecisionPolicy::Average) } {
            let x = reflect.select(Vf::ONE - z, z);

            // PadeApproximate[Ln[Gamma[x+1]], {x,5.000000001,7,9}]
            let mut y = x.poly_rational_p::<P, _, _>(
                &[
                    -6.740081381906293e-8,
                    -0.0063027,
                    -0.00313365,
                    0.00484209,
                    0.00371249,
                    0.000817098,
                    0.0000633298,
                    1.4020715520842525e-6,
                ],
                &[
                    0.0109199,
                    0.0209862,
                    0.0139389,
                    0.00397223,
                    0.000494748,
                    0.0000243068,
                    3.254907996961247e-7,
                    -6.766728779753463e-10,
                    4.8904339460457185e-12,
                    -2.5067334240332045e-14,
                ],
            );

            // since the above approximation is of lgamma(x+1), we need to offset by 1x,
            // or if reflected then by sin(pi * x) / x, which since we're in log-space
            // we take the log of below. Doing it deferred like this allows us to
            // avoid computing multiple logarithms for both cases.
            let mut e = x;

            // reflection for negative values
            if P::POLICY.avoid_branching || thermite::unlikely(reflect.any()) {
                let pix = (z * Vf::PI).sin_p::<P>();

                signum |= reflect.select(pix.signed_zero(), signum);

                e = reflect.select(pix.abs() / x, x);
                y = reflect.select(Vf::LN_PI - y, y);
            }

            y -= e.ln_p::<P>();

            return (y, signum);
        }

        let mut t = Vf::ONE;

        if P::POLICY.avoid_branching || reflect.any() {
            let pix = z.sin_pix_p::<P>();

            signum |= reflect.select(pix.signed_zero(), signum);

            t = reflect.select(pix.abs(), t);
            z = z.abs();
        }

        let b = z - Vf::HALF;
        let g = Vf::splat(LANCZOS_G);

        let mut lanczos_sum = z.poly_rational_p::<P, _, _>(&LANCZOS_P_EXPG_SCALED, &LANCZOS_Q);

        // Full A term
        let mut a = (b + g).ln_p::<P>() - Vf::ONE;

        // tiny value handling
        if const { P::POLICY.precision.gt(PrecisionPolicy::Average) } {
            let is_not_tiny = z.cmp_ge(Vf::SQRT_EPSILON);

            // shove the tiny result into the log down below
            lanczos_sum = is_not_tiny.select(lanczos_sum, z.reciprocal_p::<P>() - Vf::EGAMMA);

            // force multiplier to zero for tiny case, allowing the modified
            // lanczos sum and ln(t) to be combined for cheap
            a &= is_not_tiny.value();
        }

        let c = (lanczos_sum * t).ln_p::<P>();

        let res = a.mul_adde(b, c);

        let y = reflect.select(Vf::LN_PI - res, res);

        (y, signum)
    }

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

        let denom = if P::POLICY.precision > PrecisionPolicy::Average {
            Vf::SQRT_E / bgh.sqrt()
        } else {
            // bump up the precision a little to improve beta function accuracy
            Vf::SQRT_E * bgh.inverse_sqrt_p::<ExtraPrecision<P>>()
        };

        result *= agh_d_cgh.powf_p::<P>(a - Vf::HALF - b) * (base.powf_p::<P>(b) * denom);

        if P::POLICY.check_overflow {
            result = is_valid.select(result, Vf::NAN);
        }

        result
    }
}

const LANCZOS_G: f32 = 1.428456135094165802001953125;

const LANCZOS_P: [f32; 6] = [
    58.52061591769095910314047740215847630266,
    182.5248962595894264831189414768236280862,
    211.0971093028510041839168287718170827259,
    112.2526547883668146736465390902227161763,
    27.5192015197455403062503721613097825345,
    2.50662858515256974113978724717473206342,
];

const LANCZOS_Q: [f32; 6] = [0.0, 24.0, 50.0, 35.0, 10.0, 1.0];

const LANCZOS_P_EXPG_SCALED: [f32; 6] = [
    14.0261432874996476619570577285003839357,
    43.74732405540314316089531289293124360129,
    50.59547402616588964511581430025589038612,
    26.90456680562548195593733429204228910299,
    6.595765571169314946316366571954421695196,
    0.6007854010515290065101128585795542383721,
];
