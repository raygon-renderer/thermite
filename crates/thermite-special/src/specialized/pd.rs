use thermite::{
    math::{
        TranscendentalMathWithPolicy,
        policy::{
            DenormalBehavior, PrecisionPolicy,
            policies::{
                AveragePrecision, CheckOverflow, CmpLessPrecision, ExtraPrecision, MediumPrecision, ReferencePrecision,
                WorstPrecision,
            },
        },
        specialized::SpecializedTranscendentalMath,
    },
    prelude::*,
};

use super::*;

impl<V: FloatVectorWithBits<Element = f64>> SpecializedSpecialMath<f64> for V
where
    V: TranscendentalMathWithPolicy<Element = f64>,
    V: SpecializedTranscendentalMath<f64>,
{
    fn bessel_j<P: Policy, const N: usize>(self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn lambert_w<P: Policy>(self) -> (Self, Self) {
        // Computes both W_0(x) and W_{-1}(x) simultaneously.
        //
        // Lambert W_0(x): principal branch, defined for x >= -1/e, returns values >= -1.
        // Lambert W_{-1}(x): secondary real branch, defined for -1/e <= x < 0, returns values <= -1.
        // Both satisfy w*e^w = x.
        //
        // Uses Halley's method with piecewise initial approximations, interleaving
        // iterations for both branches to maximize instruction-level parallelism.
        // f64 needs more iterations than f32 due to 52-bit mantissa.
        //
        // Halley's iteration for w*exp(w) = x:
        //   ew = exp(w), f = w*ew - x, wp1 = w + 1
        //   Denominator rewritten to avoid an extra division:
        //     d = 2*wp1^2*ew - (w+2)*f
        //   w' = w - 2*wp1*f / d

        // For initial guess and first Halley iterations, use fast and loose precision
        type Approx<P> = WorstPrecision<CheckOverflow<P, false>>;

        let x = self;

        // --- Initial approximation (piecewise) ---
        //
        // Branch-point region (x near -1/e): damped Puiseux series.
        // See ps.rs lambert_w for full derivation.

        let p0 = x.mul_adde(Self::E, Self::ONE); // ex + 1
        let p = (p0 + p0).sqrt(); // sqrt(2(ex+1))

        // p*(1 + p*(-1/3 + p*11/72))
        let puiseux_numer = p * p.mul_adde(
            p.mul_adde(
                thermite::const_splat!(f64: 11.0 / 72.0),
                thermite::const_splat!(f64: -1.0 / 3.0),
            ),
            Self::ONE,
        );

        // 1 + K*p_0*p
        let puiseux_denom = p0.mul_adde(p * thermite::const_splat!(f64: 0.12991546098765432), Self::ONE);

        let puiseux = puiseux_numer / puiseux_denom;

        // W_0 branch: -1 + series, W_{-1} branch: -1 - series
        let w0_branch = puiseux + Self::NEG_ONE;
        let wm1_branch = Self::NEG_ONE - puiseux;

        // W_0 middle region: ex/(2+ex), exact at x = -1/e and x = 0.
        let ex = x * Self::E;
        let w0_mid = ex / (Self::TWO + ex);

        // Shared ln for asymptotic regions
        let lnx = x.abs().ln_p::<Approx<P>>();

        // W_0 asymptotic (x > e): L_1 - L_2 + L_2/L_1 where L_1 = ln(x), L_2 = ln(L_1).
        // The L_2/L_1 correction is 0 at x = e (since L_2 = ln(1) = 0), so it doesn't
        // overshoot near the transition, but closes the gap at large x.
        let l2 = lnx.ln_p::<Approx<P>>();
        let w0_asymptotic = (lnx - l2) + (l2 / lnx);

        // W_{-1} asymptotic (x near 0^-): L_1 - L_2 where L_1 = ln(-x), L_2 = ln(-L_1)
        // lnx = ln(|x|) = ln(-x) since x < 0; this is negative for small |x|.
        // -lnx is positive, so (-lnx).ln() = ln(-ln(-x)) = L_2.
        let wm1_asymptotic = lnx - (-lnx).ln_p::<Approx<P>>();

        // Select initial guesses
        let near_branch = x.cmp_lt(Self::splat(-0.1));
        let large = x.cmp_gt(Self::E);
        let mut w0 = near_branch.select(w0_branch, large.select(w0_asymptotic, w0_mid));

        let near_branch_m1 = x.cmp_lt(Self::splat(-0.25));
        let mut wm1 = near_branch_m1.select(wm1_branch, wm1_asymptotic);

        // --- Interleaved Halley iterations ---
        // Use cheap exp for warmup iteration, full-precision exp for the final ones.

        #[inline(always)]
        fn halley_step<P: Policy, W>(w: W, x: W) -> W
        where
            W: FloatVectorWithBits<Element = f64> + SpecializedTranscendentalMath<f64>,
        {
            // Use exp(-w) to avoid overflow/underflow in e^w for extreme w.
            // g = w - x*e^{-w} = f*e^{-w}, d = (w^2+2w+2) + (w+2)*x*e^{-w}
            // g and d are both single FMAs off enw, independent of each other.
            let enw = (-w).exp_p::<P>();

            let wp1 = w + W::ONE;
            let q = wp1.mul_adde(wp1, W::ONE); // (w+1)^2 + 1 = w^2 + 2w + 2
            let wp2h_x = wp1.mul_adde(x, x); // (w+2)*x - no exp dependency
            let g = x.nmul_adde(enw, w); // w - x*e^{-w}
            let d = wp2h_x.mul_adde(enw, q); // (w+2)*x*e^{-w} + (w^2+2w+2)
            (wp1 + wp1).nmul_adde(g / d, w)
        }

        #[rustfmt::skip]
        let num_iters = if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } { 3 } else { 2 };

        // warmup iteration with the looser precision to get close enough
        // for the main iterations to converge in the target precision
        w0 = halley_step::<Approx<P>, Self>(w0, x);
        wm1 = halley_step::<Approx<P>, Self>(wm1, x);

        let mut _iter = 0usize;
        while _iter < num_iters {
            _iter += 1;
            w0 = halley_step::<CheckOverflow<P, false>, Self>(w0, x);
            wm1 = halley_step::<CheckOverflow<P, false>, Self>(wm1, x);
        }

        // --- Edge cases ---
        if const { P::POLICY.precision.ge(PrecisionPolicy::Average) } {
            let x_is_zero = x.is_zero();

            // At x = -1/e, both W_0 and W_{-1} = -1
            w0 = x.cmp_eq(Self::FRAC_NEG_1_E).select(Self::NEG_ONE, w0);
            w0 = w0.nz(x_is_zero); // W_0(0) = 0

            wm1 = x.cmp_eq(Self::FRAC_NEG_1_E).select(Self::NEG_ONE, wm1);
            wm1 = x_is_zero.select(Self::NEG_INFINITY, wm1); // W_{-1}(0) = -inf
        }

        if const { matches!(P::POLICY.denormal_behavior, DenormalBehavior::Preserve) } {
            // for subnormal inputs, W_0(x) ≈ x
            w0 = x.is_subnormal().select(x, w0);
        }

        if const { P::POLICY.check_overflow } {
            let in_domain = x.cmp_ge(Self::FRAC_NEG_1_E);

            // W_0 is undefined for x < -1/e, +inf -> +inf
            w0 = in_domain.select(w0, Self::NAN);
            w0 = x.cmp_eq(Self::INFINITY).select(Self::INFINITY, w0);

            // W_{-1} is only defined for -1/e <= x < 0
            wm1 = in_domain.select(wm1, Self::NAN);
            wm1 = x.cmp_gt(Self::ZERO).select(Self::NAN, wm1);
        }

        (w0, wm1)
    }

    #[inline(always)]
    #[allow(const_item_mutation)]
    fn erf<P: Policy>(self) -> Self {
        erf_d_internal::<Self, P, false, false>(self, &mut V::EMPTY)
    }

    #[inline(always)]
    #[allow(const_item_mutation)]
    fn erfc<P: Policy>(self) -> Self {
        erf_d_internal::<Self, P, true, false>(self, &mut V::EMPTY)
    }

    #[inline(always)]
    fn lgamma<P: Policy>(self) -> Self {
        Self::lgamma_r::<P>(self).0
    }

    #[inline(always)]
    fn tgamma<P: Policy>(self) -> Self {
        todo!()
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

        // encourage instruction-level parallelism
        result *= agh_d_cgh.powf_p::<P>(a - Self::HALF - b) * base.powf_p::<P>(b) * (Self::SQRT_E / bgh.sqrt());

        if P::POLICY.check_overflow {
            result = is_valid.select(result, Self::NAN);
        }

        result
    }

    #[inline(always)]
    fn expint<P: Policy, const N: usize>(self) -> Self {
        generic::expint::expint_double::<P, f64, Self, N>(self)
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

impl<V: FloatVectorWithBits<Element = f64>> SpecializedRealSpecialMath<f64> for V
where
    V: TranscendentalMathWithPolicy<Element = f64>,
    V: SpecializedTranscendentalMath<f64>,
{
    #[inline(always)]
    fn erfinv<P: Policy>(self) -> Self {
        let y = self.flush_denormals_p::<P>();
        let a = y.abs();

        let w = -a.nmul_adde(a, V::ONE).ln_p::<P>();

        // https://www.desmos.com/calculator/yduhxx1ukm values extracted via JS console
        let mut p0 = (w - thermite::const_splat!(f64: 2.5)).poly_rev_p::<P, _>(&[
            -3.605158594283844e-12,
            -1.1526825105953649e-11,
            4.340759057762667e-10,
            2.378447620687541e-9,
            -7.498144332533493e-9,
            1.565009183876413e-8,
            4.691555466910589e-7,
            -0.000003451228003698613,
            -0.000005055953518603739,
            0.00021818504236422313,
            -0.001252754693878528,
            -0.0041773392840529855,
            0.2466402709383954,
            1.501409350414994,
        ]);

        let w_big = w.cmp_ge(thermite::const_splat!(f64: 5.0)); // at around |x| > 0.99662533231, so unlikely

        if P::POLICY.avoid_branching || thermite::unlikely(w_big.any()) {
            let mut p1 = (w.sqrt() - thermite::const_splat!(f64: 3.0)).poly_rev_p::<P, _>(&[
                -0.0000023620166848468398,
                -0.00007449590390143766,
                -0.0010722580888930223,
                -0.009288117987439485,
                -0.05369968979686224,
                -0.21681459128064842,
                -0.6192716293714041,
                -1.2294848322739875,
                -1.574375166164548,
                -0.9258061028319879,
                0.7600225853251197,
                2.3489887347568135,
                2.559965578101086,
                1.5950004257395263,
                1.5466942804733321,
                2.914513093490991,
            ]);

            if P::POLICY.check_overflow {
                p1 = a.cmp_eq(V::ONE).select(V::INFINITY, p1); // erfinv(x == 1) = inf
                p1 = a.cmp_gt(V::ONE).select(V::NAN, p1); // erfinv(x > 1) = NaN
            }

            p0 = w_big.select(p1, p0);
        }

        p0 * y
    }

    #[inline(always)]
    fn lgamma_r<P: Policy>(self) -> (Self, Self) {
        todo!()
    }

    #[inline(always)]
    fn probit<P: Policy>(self) -> Self {
        todo!()
    }

    // same form as f32
    #[inline(always)]
    fn gelu<P: Policy>(self, alpha: Self) -> (Self, Self) {
        let x = self;

        let alpha_x = alpha * x;

        // GELU(x) = 0.5 * x * (1 + erf(ax / sqrt(2)))
        let mut exp_neg_ax2 = Self::EMPTY;
        let erf = erf_d_internal::<V, P, false, true>(alpha_x * Self::FRAC_1_SQRT_2, &mut exp_neg_ax2);

        let y;
        let dy;

        let half_erf = erf.mul_adde(Self::HALF, Self::HALF);

        if V::HAS_TRUE_FMA {
            let half_x = x * Self::HALF;
            y = half_x.mul_add(erf, half_x); // fma(0.5x, erf, 0.5x), one rounding
            dy = (alpha_x * Self::FRAC_1_SQRT_TAU).mul_add(exp_neg_ax2, half_erf);
        } else {
            y = half_erf * x;
            dy = half_erf + alpha_x * Self::FRAC_1_SQRT_TAU * exp_neg_ax2;
        }

        (y, dy)
    }
}

#[rustfmt::skip]
#[inline(always)]
fn erf_d_internal<V: FloatVectorWithBits<Element = f64>, P: Policy, const C: bool, const O: bool>(x0: V, out_exp_neg_x2: &mut V) -> V {
    // Extract the sign bit once. abs(x0) = x0 ^ sign, and sign is reused
    // for the final operation in every branch, avoiding a redundant bitand.
    let sign = x0.signed_zero();
    let mut x = (x0 ^ sign).flush_denormals_p::<P>();

    // if ignoring denormals, just multiple x0 by itself to save like one cycle,
    // instead of waiting on abs(), otherwise use the denormal-flushed x value
    let x2 = if const { matches!(P::POLICY.denormal_behavior, DenormalBehavior::Ignore) } {
        x0 * x0
    } else {
        x * x
    };

    // LLVM will still start on exp and interleave it with the below operations.
    let e = (-x2).exp_p::<P>();

    let a0: V = thermite::const_splat!(f64: 0.56418958354775629);
    let a1 = x + thermite::const_splat!(f64: 2.06955023132914151);

    let b0 = x2 + x.mul_adde(thermite::const_splat!(f64: 2.71078540045147805), thermite::const_splat!(f64: 5.80755613130301624));
    let b1 = x2 + x.mul_adde(thermite::const_splat!(f64: 3.47954057099518960), thermite::const_splat!(f64: 12.06166887286239555));

    let c0 = x2 + x.mul_adde(thermite::const_splat!(f64: 3.47469513777439592), thermite::const_splat!(f64: 12.07402036406381411));
    let c1 = x2 + x.mul_adde(thermite::const_splat!(f64: 3.72068443960225092), thermite::const_splat!(f64: 8.44319781003968454));

    let d0 = x2 + x.mul_adde(thermite::const_splat!(f64: 4.00561509202259545), thermite::const_splat!(f64: 9.30596659485887898));
    let d1 = x2 + x.mul_adde(thermite::const_splat!(f64: 3.90225704029924078), thermite::const_splat!(f64: 6.36161630953880464));

    let e0 = x2 + x.mul_adde(thermite::const_splat!(f64: 5.16722705817812584), thermite::const_splat!(f64: 9.12661617673673262));
    let e1 = x2 + x.mul_adde(thermite::const_splat!(f64: 4.03296893109262491), thermite::const_splat!(f64: 5.13578530585681539));

    let f0 = x2 + x.mul_adde(thermite::const_splat!(f64: 5.95908795446633271), thermite::const_splat!(f64: 9.19435612886969243));
    let f1 = x2 + x.mul_adde(thermite::const_splat!(f64: 4.11240942957450885), thermite::const_splat!(f64: 4.48640329523408675));

    let m = if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
        // independent divisions yield slightly improved accuracy,
        // but divisions are slow, so only use for the best precision policies
        (a0 / a1) * (b0 / b1) * (c0 / c1) * (d0 / d1) * (e0 / e1) * (f0 / f1)
    } else {
        // otherwise use a single division
        let n = (a0 * b0) * (c0 * d0) * (e0 * f0);
        let d = (a1 * b1) * (c1 * d1) * (e1 * f1);
        n / d
    };

    if O {
        // write this right before we use e normally, so LLVM can interleave exp with the above
        *out_exp_neg_x2 = e;
    }

    if !C {
        e.nmul_adde(m, V::ONE) ^ sign
    } else if const { V::HAS_TRUE_FMA } {
        // exploit instruction-level parallelism if FMA is available
        x0.select_negative(m.nmul_add(e, V::TWO), m * e)
    } else {
        let y = m * e;

        x0.select_negative(V::TWO - y, y)
    }
}
