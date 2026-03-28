use thermite::{
    math::{
        TranscendentalMathWithPolicy,
        policy::{
            DenormalBehavior, PrecisionPolicy,
            policies::{
                AveragePrecision, CheckOverflow, CmpLessPrecision, ExtraPrecision, LessPrecision, MediumPrecision,
                ReferencePrecision, WorstPrecision,
            },
        },
        specialized::SpecializedTranscendentalMath,
    },
    prelude::*,
};

use super::*;

impl<V: FloatVectorWithBits<Element = f32>> SpecializedSpecialMath<f32> for V
where
    V: TranscendentalMathWithPolicy<Element = f32>,
    V: SpecializedTranscendentalMath<f32>,
{
    #[inline(always)]
    fn lambert_w<P: Policy>(self) -> (Self, Self) {
        // Computes both W₀(x) and W₋₁(x) simultaneously.
        //
        // Lambert W₀(x): principal branch, defined for x >= -1/e, returns values >= -1.
        // Lambert W₋₁(x): secondary real branch, defined for -1/e <= x < 0, returns values <= -1.
        // Both satisfy w·eʷ = x.
        //
        // Uses Halley's method with piecewise initial approximations, interleaving
        // iterations for both branches to maximize instruction-level parallelism.
        //
        // Halley's iteration for w·exp(w) = x:
        //   ew = exp(w), f = w·ew - x, wp1 = w + 1
        //   Denominator rewritten to avoid an extra division:
        //     d = 2·wp1²·ew - (w+2)·f
        //   w' = w - 2·wp1·f / d

        // For initial guess and first Halley iterations, use fast and loose precision
        type Approx<P> = WorstPrecision<CheckOverflow<P, false>>;

        let x = self;

        // --- Initial approximation (piecewise) ---
        //
        // Branch-point region (x near -1/e): damped Puiseux series.
        //
        // W has a square-root singularity at x = -1/e (double root of w·eʷ - x at w = -1),
        // so Halley degenerates to linear convergence without a sqrt-based initial guess.
        //
        // The raw Puiseux series is W ≈ -1 ± (p - p²/3 + 11p³/72) where p = sqrt(2(ex+1)),
        // with + for W₀ and - for W₋₁. This converges well near -1/e but diverges further
        // out. We damp it with a denominator that grows with distance from -1/e:
        //
        //   w_branch = -1 ± p·(1 + p·(-1/3 + p·11/72)) / (1 + K·p₀·p)
        //
        // where p₀ = ex+1, and K = 1/(C · e^(3/2) · √2) with C ≈ 1.2144578338 found by
        // minimizing the integrated backward error |w·eʷ - x| over [-1/e, 0] in Desmos.
        // The denominator arises from (x + 1/e)^1.5 / C = (p₀/e)^1.5 / C = p₀·p / (C·e^(3/2)·√2).

        let p0 = x.mul_adde(Self::E, Self::ONE); // ex + 1
        let p = (p0 + p0).sqrt(); // sqrt(2(ex+1))

        // p·(1 + p·(-1/3 + p·11/72))
        let puiseux_numer = p * p.mul_adde(
            p.mul_adde(
                thermite::generic_splat!(f32: 11.0 / 72.0),
                thermite::generic_splat!(f32: -1.0 / 3.0),
            ),
            Self::ONE,
        );

        // 1 + K·p₀·p
        let puiseux_denom = p0.mul_adde(p * thermite::generic_splat!(f32: 0.12991546098765432), Self::ONE);

        let puiseux = puiseux_numer / puiseux_denom;

        // W₀ branch: -1 + series, W₋₁ branch: -1 - series
        let w0_branch = puiseux + Self::NEG_ONE;
        let wm1_branch = Self::NEG_ONE - puiseux;

        // W₀ middle region: ex/(2+ex)
        let ex = x * Self::E;
        let w0_mid = ex / (Self::TWO + ex);

        // Shared ln for asymptotic regions
        let lnx = x.abs().ln_p::<Approx<P>>();

        // W₀ asymptotic (x > e): L₁ - L₂ + L₂/L₁ where L₁ = ln(x), L₂ = ln(L₁).
        // The L₂/L₁ correction is 0 at x = e (since L₂ = ln(1) = 0), so it doesn't
        // overshoot near the transition, but closes the gap at large x.
        let l2 = lnx.ln_p::<Approx<P>>();
        let l2_corr = if const { P::POLICY.precision.le(PrecisionPolicy::Average) } {
            l2 * lnx.reciprocal_p::<Approx<P>>()
        } else {
            l2 / lnx
        };
        let w0_asymptotic = lnx - l2 + l2_corr;

        // W₋₁ asymptotic (x near 0⁻): L₁ - L₂ where L₁ = ln(-x), L₂ = ln(-L₁)
        // lnx = ln(|x|) = ln(-x) since x < 0; this is negative for small |x|.
        // -lnx is positive, so (-lnx).ln() = ln(-ln(-x)) = L₂.
        let wm1_asymptotic = lnx - (-lnx).ln_p::<Approx<P>>();

        // Select initial guesses
        let near_branch = x.cmp_lt(V::splat(-0.1));
        let large = x.cmp_gt(Self::E);

        let mut w0 = near_branch.select(w0_branch, large.select(w0_asymptotic, w0_mid));

        let near_branch_m1 = x.cmp_lt(V::splat(-0.25));
        let mut wm1 = near_branch_m1.select(wm1_branch, wm1_asymptotic);

        // --- Interleaved Halley iterations ---
        #[inline(always)]
        fn halley_step<P: Policy, W>(w: W, x: W) -> W
        where
            W: FloatVectorWithBits<Element = f32> + SpecializedTranscendentalMath<f32>,
        {
            let ew = w.exp_p::<P>();
            let f = w.mul_sube(ew, x);
            let wp1 = w + W::ONE;
            let wp2 = wp1 + wp1;
            let d = (wp1 + W::ONE).nmul_adde(f, wp2 * wp1 * ew);
            wp2.nmul_adde(f / d, w)
        }

        #[rustfmt::skip]
        let num_iters = if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } { 2 } else { 1 };

        // warmup iteration with the looser precision to get close enough
        // for the main iterations to converge in the target precision
        w0 = halley_step::<Approx<P>, Self>(w0, x);
        wm1 = halley_step::<Approx<P>, Self>(wm1, x);

        for _ in 0..num_iters {
            // don't need to check overflow within these since it should be well-defined for
            // all intermediate values, and the final check will catch any issues.
            w0 = halley_step::<CheckOverflow<P, false>, Self>(w0, x);
            wm1 = halley_step::<CheckOverflow<P, false>, Self>(wm1, x);
        }

        // --- Edge cases ---
        if const { P::POLICY.precision.ge(PrecisionPolicy::Average) } {
            let x_is_zero = x.is_zero();

            // At x = -1/e, both W₀ and W₋₁ = -1
            w0 = x.cmp_eq(Self::FRAC_NEG_1_E).select(Self::NEG_ONE, w0);
            // Honestly the approximation handles W₀(0) = 0 pretty well,
            // but just in case, explicitly set it to the correct value.
            w0 = w0.nz(x_is_zero); // W₀(0) = 0

            wm1 = x.cmp_eq(Self::FRAC_NEG_1_E).select(Self::NEG_ONE, wm1);
            wm1 = x_is_zero.select(Self::NEG_INFINITY, wm1); // W₋₁(0) = -inf
        }

        if const { P::POLICY.check_overflow } {
            let in_domain = x.cmp_ge(Self::FRAC_NEG_1_E);

            // W₀ is undefined for x < -1/e, +inf -> +inf
            w0 = in_domain.select(w0, Self::NAN);
            w0 = x.cmp_eq(Self::INFINITY).select(Self::INFINITY, w0);

            // W₋₁ is only defined for -1/e <= x < 0
            wm1 = in_domain.select(wm1, Self::NAN);
            wm1 = x.cmp_gt(Self::ZERO).select(Self::NAN, wm1);
        }

        (w0, wm1)
    }

    #[inline(always)]
    fn erf<P: Policy>(self) -> Self {
        let x0 = self;

        if const { P::POLICY.precision.eq(PrecisionPolicy::Reference) } {
            // Use erf(x) = 1 - erfc(x), since erfc has a good reference implementation
            return V::ONE - x0.erfc_p::<P>();
        }

        let mut x = x0.abs().flush_denormals_p::<P>();

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

        let x = x0.abs().flush_denormals_p::<P>();
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
        let x = self
            .flush_denormals_p::<P>()
            .clamp(V::splat(-0.99999), V::splat(0.99999));

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
    fn sigmoid<P: Policy>(self) -> Self {
        if const { P::POLICY.precision.gt(PrecisionPolicy::Average) } {
            let is_pos = self.is_positive();
            let x = self.neg_c(is_pos); // conditionally negate if positive
            let e = x.exp_p::<P>();

            let n = is_pos.select(Self::ONE, e);
            let d = Self::ONE + e;

            return n / d;
        }

        // exp at Medium is the same, exp at Worst is much faster and still accurate enough,
        // so use the faster one for HighPerformance policy.
        (Self::ONE + (-self).exp_p::<LessPrecision<P>>()).reciprocal_p::<ExtraPrecision<P>>()
    }

    // This ended up being a bust, but I'll keep it around anyway.
    // #[inline(always)]
    // fn sigmoid<P: Policy>(self) -> Self {
    //     if const { P::POLICY.precision.ge(PrecisionPolicy::Average) } {
    //         Self::ONE / (Self::ONE + (-self).exp_p::<P>())
    //     } else {
    //         let (r, d) = const {
    //             match P::POLICY.precision {
    //                 PrecisionPolicy::Worst => (8, -1.0 / (1 << 8) as f32),
    //                 PrecisionPolicy::Medium => (12, -1.0 / (1 << 12) as f32),
    //                 _ => (0, 0.0), // not used since Average and above use the other method
    //             }
    //         };

    //         let mut base = self.mul_adde(Self::splat(d), Self::ONE);

    //         for _ in 0..(r - 1) {
    //             base *= base;
    //         }

    //         base.mul_adde(base, Self::ONE).reciprocal_p::<P>()
    //     }
    // }

    #[inline(always)]
    fn tgamma<P: Policy>(z: Self) -> Self {
        if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
            // We have a good lgamma approximation, so use it for tgamma on lower precisions.
            let (lgamma, sign) = z.lgamma_r_p::<P>();

            // use min(P + 1, Average) precision here. We want decent precision,
            // but not more than average.
            return lgamma.exp_p::<ExtraPrecision<P>>() * sign;
        }

        let mut z = z.flush_denormals_p::<P>();

        let orig_z = z;

        let is_negative = z.is_negative();
        let mut reflected = GenericMask::FALSY;

        let mut res = Self::ONE;

        // Reflect ALL negative values via Γ(z) = -π / (z·sin(πz)·Γ(|z|))
        // This avoids the repeated-division recurrence which accumulates rounding error.
        if P::POLICY.avoid_branching || is_negative.any() {
            reflected = is_negative;
            let refl_res = z * z.sin_pi_p::<P>(); // z · sin(πz)
            res = reflected.select(refl_res, res);
            z = z.abs();
        }

        // Negative integer poles and ±0
        let is_neg_int = is_negative & orig_z.cmp_eq(orig_z.floor()) & orig_z.cmp_ne(Self::ZERO);
        let is_zero = orig_z.cmp_eq(Self::ZERO);

        // Shift z ∈ (SQRT_EPSILON, 1) up by 1 via Γ(z) = Γ(z+1)/z.
        // The Lanczos polynomial is fit for z >= 1; evaluating below that is the
        // primary source of error in the (0, 1) range.
        if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            let needs_shift = z.cmp_lt(Self::ONE) & z.cmp_ge(Self::SQRT_EPSILON);
            res = needs_shift.select(res / z, res);
            z = needs_shift.select(z + Self::ONE, z);
        }

        // Integers (positive, after reflection)

        let mut is_int = GenericMask::FALSY;
        let mut int_res = Self::ONE;

        if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            let zf = z.floor();
            // Cap at 36 - Γ overflows f32 beyond that, and this bounds the loop.
            is_int = zf.cmp_eq(z) & zf.cmp_lt(Self::splat(36.0)) & !is_neg_int & !is_zero;

            if thermite::unlikely(is_int.any()) {
                let mut j = Self::ONE;
                // Mask with is_int so non-integer lanes with large zf can't keep the loop alive.
                let mut k = j.cmp_lt(zf) & is_int;

                while k.any() {
                    int_res = k.select(int_res * j, int_res);
                    j += Self::ONE;
                    k = j.cmp_lt(zf) & is_int;
                }

                if thermite::unlikely(is_int.all()) {
                    return int_res;
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
            let tiny_res = z.reciprocal_p::<P>() - Self::EULER_GAMMA;
            res *= is_tiny.select(tiny_res, normal_res);
        } else {
            res *= normal_res;
        }

        // Edge cases: Γ(-int) = NaN, Γ(±0) = ±∞
        let zero_res = is_negative.select(Self::NEG_INFINITY, Self::INFINITY);
        let result = reflected.select(-Self::PI / res, is_int.select(int_res, res));
        let mut result = is_neg_int.select(Self::NAN, result);

        if const {
            P::POLICY.precision.ge(PrecisionPolicy::Best)
                && matches!(P::POLICY.denormal_behavior, DenormalBehavior::Preserve)
        } {
            let is_subnormal = z.is_subnormal();

            if thermite::unlikely(is_subnormal.any()) {
                result = is_subnormal.select(Self::ONE / orig_z, result);
            }
        }

        is_zero.select(zero_res, result)
    }

    #[inline(always)]
    fn lgamma_r<P: Policy>(z: Self) -> (Self, Self) {
        let mut z = z.flush_denormals_p::<P>();
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
            lanczos_sum = is_not_tiny.select(lanczos_sum, z.reciprocal_p::<P>() - Self::EULER_GAMMA);

            // force multiplier to zero for tiny case, allowing the modified
            // lanczos sum and ln(t) to be combined for cheap
            a = a.z(is_not_tiny);
        }

        let c = (lanczos_sum * t).ln_p::<P>();

        let res = a.mul_adde(b, c);

        let y = reflect.select(Self::LN_PI - res, res);

        (y, signum)
    }

    #[inline(always)]
    fn beta<P: Policy>(a: Self, b: Self) -> Self {
        let (a, b) = (a.flush_denormals_p::<P>(), b.flush_denormals_p::<P>());

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

    #[inline(always)]
    fn expint<P: Policy, const N: usize>(self) -> Self {
        generic::expint::expint_generic::<P, f32, Self, N>(self)
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
