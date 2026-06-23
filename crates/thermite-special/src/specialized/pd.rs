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

use crate::RealSpecialMathWithPolicy as _;

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
        let z = self;

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

        // Reflect ALL negative values via Γ(z) = -π / (z*sin(πz)*Γ(|z|))
        // This avoids the repeated-division recurrence which accumulates rounding error.
        if const { P::POLICY.avoid_branching } || is_negative.any() {
            reflected = is_negative;
            let refl_res = z * z.sin_pi_p::<P>(); // z * sin(πz)
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
            // Cap at 172 - Γ overflows f64 beyond that, and this bounds the loop.
            is_int = zf.cmp_eq(z) & zf.cmp_lt(Self::splat(172.0)) & !is_neg_int & !is_zero;

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

        let gh = Self::splat(const { LANCZOS_G - 0.5 });

        // Uses the leading-term-first (reversed) Lanczos arrays - see LANCZOS_P_REV.
        let lanczos_sum = z.poly_rev_p::<P, _>(&LANCZOS_P_REV) / z.poly_rev_p::<P, _>(&LANCZOS_Q_REV);

        let zgh = z + gh;
        let lzgh = zgh.ln_p::<P>();

        // (z * lzfg) > ln(f64::MAX)
        let very_large = (z * lzgh).cmp_gt(Self::splat(709.782712893383973096206318586483));

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
    fn digamma<P: Policy>(self) -> Self {
        // Asymptotic expansion coefficients for x >= 10 (17-digit precision, 53-bit mantissa).
        // Coefficients from Boost.Math digamma_imp_large (BSL-1.0).
        const P_LARGE: [f64; 8] = [
            0.083333333333333333333333333333333333333333333333333,
            -0.0083333333333333333333333333333333333333333333333333,
            0.003968253968253968253968253968253968253968253968254,
            -0.0041666666666666666666666666666666666666666666666667,
            0.0075757575757575757575757575757575757575757575757576,
            -0.021092796092796092796092796092796092796092796092796,
            0.083333333333333333333333333333333333333333333333333,
            -0.44325980392156862745098039215686274509803921568627,
        ];

        // Rational approximation on [1, 2]: digamma(x) = (x - root) * (Y + R(x-1)).
        // 18-digit precision (53-bit mantissa). Coefficients from Boost.Math
        // digamma_imp_1_2 (BSL-1.0).
        // root = ROOTS[0] + ROOTS[1] + ROOTS[2], summed via staged subtraction for bits.
        const Y: f64 = 0.99558162689208984;
        const ROOTS: [f64; 3] = [
            1569415565.0 / 1073741824.0,                 // / 2^30
            (381566830.0 / 1073741824.0) / 1073741824.0, // / 2^60
            0.9016312093258695918615325266959189453125e-19,
        ];
        const P_12: [f64; 6] = [
            0.25479851061131551,
            -0.32555031186804491,
            -0.65031853770896507,
            -0.28919126444774784,
            -0.045251321448739056,
            -0.0020713321167745952,
        ];
        const Q_12: [f64; 7] = [
            1.0,
            2.0767117023730469,
            1.4606242909763515,
            0.43593529692665969,
            0.054151797245674225,
            0.0021284987017821144,
            -0.55789841321675513e-6,
        ];

        generic::digamma::digamma_impl::<P, _, _, _, _, _, _>(self, Y, &ROOTS, &P_LARGE, &P_12, &Q_12)
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

// `tgamma` evaluates the unscaled Lanczos sum with `poly_rev_p` (Horner from the
// leading coefficient), which often optimizes better. `poly_rev_p` wants
// leading-term-first order, so these are the canonical (constant-term-first)
// arrays written out in reverse. `LANCZOS_Q` below is kept in constant-term-first
// order for `lgamma_r`/`beta`, which consume it through `poly_rational_p`.
const LANCZOS_P_REV: [f64; 13] = [
    2.506628274631000270164908177133837338626,
    210.8242777515793458725097339207133627117,
    8071.672002365816210638002902272250613822,
    186056.2653952234950402949897160456992822,
    2876370.628935372441225409051620849613599,
    31426415.58540019438061423162831820536287,
    248874557.8620541565114603864132294232163,
    1439720407.311721673663223072794912393972,
    6039542586.352028005064291644307297921070,
    17921034426.03720969991975575445893111267,
    35711959237.35566804944018545154716670596,
    42919803642.64909876895789904700198885093,
    23531376880.41075968857200767445163675473,
];

const LANCZOS_Q_REV: [f64; 13] = [
    1.0,
    66.0,
    1925.0,
    32670.0,
    357423.0,
    2637558.0,
    13339535.0,
    45995730.0,
    105258076.0,
    150917976.0,
    120543840.0,
    39916800.0,
    0.0,
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
        // Branchless erfinv: a cheap Winitzki seed refined with Halley iterations
        // against the (accurate) erfc, which is far friendlier to SIMD than the
        // many-branch piecewise-rational approach.
        //
        // We solve erfc(x) = q for x >= 0, where q = 1 - |y|. The Newton/Halley
        // residual erf(x) - |y| is evaluated as q - erfc(x): in the tail both
        // terms are tiny, so their difference keeps full relative precision (the
        // direct form erf(x) - |y| would cancel two ~1 values down to noise).
        // Halley is cubic, so the ~1% Winitzki seed reaches full f64 in 2 steps.
        const ALPHA: f64 = 0.147;
        const RCP_PI_ALPHA_2: f64 = 4.330746750799873; // 2 / (pi * ALPHA)
        const RCP_ALPHA: f64 = 1.0 / ALPHA;
        const SQRT_PI_2: f64 = 0.8862269254527580136490837416706; // sqrt(pi) / 2 = 1 / erf'(0)

        let y = self.flush_denormals_p::<P>();
        let a = y.abs();
        let q = Self::ONE - a; // 1 - |y|
        let omsq = q * (Self::ONE + a); // 1 - y^2, computed without cancellation near |y| = 1

        // Winitzki seed (magnitude): sqrt(sqrt(t1^2 - ln(1-y^2)/alpha) - t1)
        let lnv = omsq.ln_p::<P>(); // ln(1 - y^2) <= 0
        let t1 = lnv.mul_adde(Self::HALF, Self::splat(RCP_PI_ALPHA_2));
        let mut x = (t1.mul_adde(t1, lnv * Self::splat(-RCP_ALPHA)).sqrt() - t1).sqrt();

        // Halley refinement: x -= u / (1 + x*u), u = (erf(x) - |y|) / erf'(x)
        //   erf(x) - |y| = q - erfc(x),   1/erf'(x) = (sqrt(pi)/2) * exp(x^2)
        let steps = if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            2
        } else {
            1
        };
        let mut i = 0;
        while i < steps {
            // erfc(x) already computes exp(-x^2); reuse it so exp(x^2) is just a reciprocal.
            let mut exp_neg = Self::EMPTY;
            let erfc = erf_d_internal::<Self, P, true, true>(x, &mut exp_neg);
            let u = (q - erfc) * Self::splat(SQRT_PI_2) / exp_neg;
            x -= u / x.mul_adde(u, Self::ONE);
            i += 1;
        }

        let mut res = x.copysign(y);

        if const { P::POLICY.check_overflow } {
            res = a.cmp_eq(Self::ONE).select(Self::INFINITY.copysign(y), res); // erfinv(+-1) = +-inf
            res = a.cmp_gt(Self::ONE).select(Self::NAN, res); // out of domain
        }

        res
    }

    #[inline(always)]
    fn lgamma_r<P: Policy>(self) -> (Self, Self) {
        let mut z = self.flush_denormals_p::<P>();
        let mut signum = Self::ONE;

        let reflect = z.is_negative();

        let mut t = Self::ONE;

        if const { P::POLICY.avoid_branching } || reflect.any() {
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
            a = a.zz(is_not_tiny);
        }

        let c = (lanczos_sum * t).ln_p::<P>();

        let res = a.mul_adde(b, c);

        let y = reflect.select(Self::LN_PI - res, res);

        (y, signum)
    }

    /// Uses the algorithm from Peter John Acklam, sourced from here:
    /// <https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/>
    #[inline(always)]
    fn probit<P: Policy>(self) -> Self {
        const A: [f64; 6] = [
            2.506628277459239e+00,
            -3.066479806614716e+01,
            1.383577518672690e+02,
            -2.759285104469687e+02,
            2.209460984245205e+02,
            -3.969683028665376e+01,
        ];
        const B: [f64; 6] = [
            1.0,
            -1.328068155288572e+01,
            6.680131188771972e+01,
            -1.556989798598866e+02,
            1.615858368580409e+02,
            -5.447609879822406e+01,
        ];
        const C: [f64; 6] = [
            2.938163982698783e+00,
            4.374664141464968e+00,
            -2.549732539343734e+00,
            -2.400758277161838e+00,
            -3.223964580411365e-01,
            -7.784894002430293e-03,
        ];
        const D: [f64; 5] = [
            1.0,
            3.754408661907416e+00,
            2.445134137142996e+00,
            3.224671290700398e-01,
            7.784695709041462e-03,
        ];

        // f64: refine the Acklam estimate with one Halley step (REFINE = true).
        generic::probit::probit_acklam::<P, _, _, true>(self, &A, &B, &C, &D)
    }

    // same form as f32
    #[inline(always)]
    fn gelu<P: Policy>(self, alpha: Self) -> Self {
        let x = self;

        let alpha_x = alpha * x;

        // GELU(x) = 0.5 * x * (1 + erf(ax / sqrt(2)))
        // O = false: skip the exp(-ax^2) byproduct that only the derivative needs.
        let mut unused = Self::EMPTY;
        let erf = erf_d_internal::<V, P, false, false>(alpha_x * Self::FRAC_1_SQRT_2, &mut unused);

        if V::HAS_TRUE_FMA {
            let half_x = x * Self::HALF;
            half_x.mul_add(erf, half_x) // fma(0.5x, erf, 0.5x), one rounding
        } else {
            erf.mul_adde(Self::HALF, Self::HALF) * x
        }
    }
}

impl<V: FloatVectorWithBits<Element = f64>> SpecializedRealPrimalMath<f64> for V
where
    V: TranscendentalMathWithPolicy<Element = f64>,
    V: SpecializedTranscendentalMath<f64>,
{
    #[inline(always)]
    fn gelu_d<P: Policy>(self, alpha: Self) -> (Self, Self) {
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
