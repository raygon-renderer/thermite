use thermite::{
    generic::GenericMask,
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

impl<V> SpecializedSpecialMath<f32> for V
where
    V: TranscendentalMathWithPolicy<Element = f32>,
    V: SpecializedTranscendentalMath<f32>,
{
    #[inline(always)]
    fn erf<P: Policy>(self) -> Self {
        let x0 = self;

        if const { P::POLICY.precision.eq(PrecisionPolicy::Reference) } {
            // Use erf(x) = 1 - erfc(x), since erfc has a good reference implementation
            return V::ONE - x0.erfc_p::<P>();
        }

        let mut x = x0.abs();

        if P::POLICY.check_overflow {
            x = V::ONE - (V::ONE - x); // crush denormals
        }

        let y = match P::POLICY.precision {
            // 5 * 10^-4 accuracy
            PrecisionPolicy::Worst => {
                let t = x.poly_p::<P, _>(&[1.0, 0.278393, 0.230389, 0.000972, 0.078108]);
                let t2 = t * t;
                let t4 = t2 * t2;

                // 1 - 1/t4
                if V::HAS_APPROX_RCP {
                    // use raw approximate reciprocal when available
                    let y = t4.rcp();

                    // combine one 1/x newton iteration with 1-y for the final result
                    y.nmul_adde(t4.nmul_adde(y, V::TWO), V::ONE)
                } else {
                    // otherwise just do the reciprocal normally
                    V::ONE - t4.reciprocal_p::<P>()
                }
            }

            // 3 * 10^-7 accuracy
            PrecisionPolicy::Medium | PrecisionPolicy::Average => {
                let r = x.poly_p::<P, _>(&[
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

                V::ONE - r16.reciprocal_p::<ExtraPrecision<P>>()
            }

            // this method is not used, but kept for reference since it's _supposedly_ more accurate,
            // but doesn't seem to be, maybe due to the reliance on the exponential function?
            // // 1.5 * 10^-7 accuracy
            // PrecisionPolicy::Average => {
            //     // 1 / (1 + p * x) where p = 0.3275911
            //     let t = x.mul_adde(V::splat(0.3275911), V::ONE).reciprocal_p::<P>();
            //
            //     let e = t * (-x * x).exp_p::<P>(); // e^(-x^2)
            //
            //     let y = t.poly_p::<P, _>(&[0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]);
            //
            //     y.nmul_adde(e, V::ONE)
            // }

            // 1.2 * 10^-7 accuracy
            PrecisionPolicy::Best => {
                // 1 / (1 + 1/2|x|)
                let t = x.mul_adde(V::HALF, V::ONE).reciprocal_p::<P>();

                let r0 = t.poly_p::<P, _>(&[
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
                t.nmul_adde(x.nmul_adde(x, r0).exp_p::<P>(), V::ONE)
            }
            PrecisionPolicy::Reference => unreachable!("Reference precision handled above"),
        };

        y.mul_sign(x0)
    }

    #[inline(always)]
    fn erfc<P: Policy>(self) -> Self {
        let x0 = self;

        if const { P::POLICY.precision.lt(PrecisionPolicy::Reference) } {
            // Use erfc(x) = 1 - erf(x)
            return V::ONE - x0.erf_p::<P>();
        }

        let x = x0.abs();
        let x2 = x0 * x0;

        let a0 = V::splat(0.56418958354775629);
        let a1 = x + V::splat(2.06955023132914151);

        let b0 = x2 + x.mul_adde(V::splat(2.06955023132914151), V::splat(5.80755613130301624));
        let b1 = x2 + x.mul_adde(V::splat(3.47954057099518960), V::splat(12.06166887286239555));

        let c0 = x2 + x.mul_adde(V::splat(3.47469513777439592), V::splat(12.07402036406381411));
        let c1 = x2 + x.mul_adde(V::splat(3.72068443960225092), V::splat(8.44319781003968454));

        let d0 = x2 + x.mul_adde(V::splat(4.00561509202259545), V::splat(9.30596659485887898));
        let d1 = x2 + x.mul_adde(V::splat(3.90225704029924078), V::splat(6.36161630953880464));

        let e0 = x2 + x.mul_adde(V::splat(5.16722705817812584), V::splat(9.12661617673673262));
        let e1 = x2 + x.mul_adde(V::splat(4.03296893109262491), V::splat(5.13578530585681539));

        let f0 = x2 + x.mul_adde(V::splat(5.95908795446633271), V::splat(9.19435612886969243));
        let f1 = x2 + x.mul_adde(V::splat(4.11240942957450885), V::splat(4.48640329523408675));

        let n = a0 * b0 * c0 * d0 * e0 * f0;
        let d = a1 * b1 * c1 * d1 * e1 * f1;

        let m = n / d;
        let e = (-x2).exp_p::<P>();

        // if x<0 then 2 - y, else y
        if V::HAS_TRUE_FMA {
            // exploit instruction-level parallelism if FMA is available
            x0.select_negative(m.nmul_add(e, V::TWO), m * e)
        } else {
            let y = m * e;

            x0.select_negative(V::TWO - y, y)
        }
    }

    #[inline(always)]
    fn erfinv<P: Policy>(self) -> Self {
        // (-1, 1) range
        let x = self.clamp(V::splat(-0.99999), V::splat(0.99999));

        let w = -x.nmul_adde(x, V::ONE).ln_p::<P>();

        let ge5 = w.cmp_ge(V::splat(5.0));

        let w0 = w - V::splat(2.5);
        let mut p0 = w0.poly_p::<P, _>(&[
            1.50140941,
            0.246640727,
            -0.00417768164,
            -0.00125372503,
            0.00021858087,
            -4.39150654e-06,
            -3.5233877e-06,
            3.43273939e-07,
            2.81022636e-08,
        ]);

        if P::POLICY.avoid_branching || thermite::unlikely(ge5.any()) {
            let w1 = w.sqrt() - V::splat(3.0);
            let p1 = w1.poly_p::<P, _>(&[
                2.83297682,
                1.00167406,
                0.00943887047,
                -0.0076224613,
                0.00573950773,
                -0.00367342844,
                0.00134934322,
                0.000100950558,
                -0.000200214257,
            ]);

            p0 = ge5.select(p1, p0);
        }

        p0 * x
    }

    #[inline(always)]
    fn tgamma<P: Policy>(mut z: Self) -> Self {
        if const { P::POLICY.precision.le(PrecisionPolicy::Average) } {
            // We have a good lgamma approximation, so use it for tgamma on lower precisions.
            let (lgamma, sign) = z.lgamma_r_p::<P>();

            // use min(P + 1, Average) precision here. We want decent precision,
            // but not more than average.
            return lgamma.exp_p::<ExtraPrecision<P>>() * sign;
        }

        let orig_z = z;

        let is_negative = z.is_negative();
        let mut reflected = GenericMask::FALSY;

        let mut res = Self::ONE;

        #[allow(clippy::never_loop)]
        'goto_positive: while is_negative.any() {
            reflected = z.cmp_le(Self::splat(-20.0));

            let mut refl_res = Self::EMPTY;

            if P::POLICY.avoid_precision_branches() || thermite::unlikely(reflected.any()) {
                refl_res = z * z.sin_pi_p::<P>(); // z * sin(pi * z)

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
                mod_z += Self::ONE & is_neg.value();
                is_neg = mod_z.is_negative();
            }

            z = reflected.select(-z, mod_z);
            res = reflected.select(refl_res, res);

            break 'goto_positive;
        }

        // Integers

        let mut is_int = GenericMask::FALSY;
        let mut int_res = Self::ONE;

        if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            let zf = z.floor();
            is_int = zf.cmp_eq(z);

            if thermite::unlikely(is_int.any()) {
                let mut j = Self::ONE;
                let mut k = j.cmp_lt(zf);

                while k.any() {
                    int_res = k.select(int_res * j, int_res);
                    j += Self::ONE;
                    k = j.cmp_lt(zf);
                }

                // Γ(-int) = NaN for poles
                int_res = is_negative.select(Self::NAN, int_res);

                // approaching zero from either side results in +/- infinity
                int_res = orig_z
                    .cmp_eq(Self::ZERO)
                    .select(is_negative.select(Self::NEG_INFINITY, Self::INFINITY), int_res);

                if thermite::unlikely(is_int.all()) {
                    return int_res; // can skip full gamma calculation if all inputs are integers
                }
            }
        }

        // Full

        let gh = Self::splat(LANCZOS_G - 0.5);

        let lanczos_sum = z.poly_rational_p::<P, _, _>(&LANCZOS_P, &LANCZOS_Q);

        let zgh = z + gh;
        let lzgh = zgh.ln_p::<P>();

        // (z * lzfg) > ln(f32::MAX)
        let very_large = (z * lzgh).cmp_gt(Self::splat(
            88.722839053130621324601674778549183073943430402325230485234240247,
        ));

        // only compute powf once
        let h = zgh.powf_p::<P>(very_large.select(z.mul_sube(Self::HALF, Self::splat(0.25)), z - Self::HALF));

        // save a couple cycles by avoiding this division, but worst-case precision is slightly worse
        let denom = if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            lanczos_sum / zgh.exp_p::<P>()
        } else {
            lanczos_sum * (-zgh).exp_p::<P>()
        };

        let normal_res = very_large.select(h * h, h) * denom;

        // Tiny
        if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            let is_tiny = z.cmp_lt(Self::SQRT_EPSILON);
            let tiny_res = z.reciprocal_p::<P>() - Self::EGAMMA;
            res *= is_tiny.select(tiny_res, normal_res);
        } else {
            res *= normal_res;
        }

        reflected.select(-Self::PI / res, is_int.select(int_res, res))
    }

    #[inline(always)]
    fn lgamma_r<P: Policy>(mut z: Self) -> (Self, Self) {
        let mut signum = Self::ONE;

        let reflect = z.is_negative();

        if const { P::POLICY.precision.le(PrecisionPolicy::Average) } {
            let x = reflect.select(Self::ONE - z, z);

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
                let pix = (z * Self::PI).sin_p::<P>();

                signum |= reflect.select(pix.signed_zero(), signum);

                e = reflect.select(pix.abs() / x, x);
                y = reflect.select(Self::LN_PI - y, y);
            }

            y -= e.ln_p::<P>();

            return (y, signum);
        }

        let mut t = Self::ONE;

        if P::POLICY.avoid_branching || reflect.any() {
            let pix = z * z.sin_pi_p::<P>(); // z * sin(pi * z)

            signum |= reflect.select(pix.signed_zero(), signum);

            t = reflect.select(pix.abs(), t);
            z = z.abs();
        }

        let b = z - Self::HALF;
        let g = Self::splat(LANCZOS_G);

        let mut lanczos_sum = z.poly_rational_p::<P, _, _>(&LANCZOS_P_EXPG_SCALED, &LANCZOS_Q);

        // Full A term
        let mut a = (b + g).ln_p::<P>() - Self::ONE;

        // tiny value handling
        if const { P::POLICY.precision.gt(PrecisionPolicy::Average) } {
            let is_not_tiny = z.cmp_ge(Self::SQRT_EPSILON);

            // shove the tiny result into the log down below
            lanczos_sum = is_not_tiny.select(lanczos_sum, z.reciprocal_p::<P>() - Self::EGAMMA);

            // force multiplier to zero for tiny case, allowing the modified
            // lanczos sum and ln(t) to be combined for cheap
            a &= is_not_tiny.value();
        }

        let c = (lanczos_sum * t).ln_p::<P>();

        let res = a.mul_adde(b, c);

        let y = reflect.select(Self::LN_PI - res, res);

        (y, signum)
    }

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

        let denom = if P::POLICY.precision > PrecisionPolicy::Average {
            Self::SQRT_E / bgh.sqrt()
        } else {
            // bump up the precision a little to improve beta function accuracy
            Self::SQRT_E * bgh.inverse_sqrt_p::<ExtraPrecision<P>>()
        };

        result *= agh_d_cgh.powf_p::<P>(a - Self::HALF - b) * (base.powf_p::<P>(b) * denom);

        if P::POLICY.check_overflow {
            result = is_valid.select(result, Self::NAN);
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
