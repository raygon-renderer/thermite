use thermite::{
    math::{
        TranscendentalMathWithPolicy,
        policy::{
            DenormalBehavior, PrecisionPolicy,
            policies::{
                CheckOverflow, ExtraPrecision, MediumPrecision, WorstPrecision,
            },
        },
        specialized::SpecializedTranscendentalMath,
    },
    prelude::*,
    register::NativeCapability,
};

use crate::RealSpecialMathWithPolicy as _;

use super::*;

impl<V: FloatVectorWithBits<Element = f32>> SpecializedSpecialMath<f32> for V
where
    V: TranscendentalMathWithPolicy<Element = f32>,
    V: SpecializedTranscendentalMath<f32>,
{
    type ExpIntDetails = Self;

    // TEMP(bessel_j): disabled until orders beyond J_0 exist - see the note in lib.rs.
    // The `bessel_j0`/`bessel_j0_pqzero` machinery this called is kept below under the
    // same marker.
    //#[inline(always)]
    //fn bessel_j<P: Policy, const N: usize>(self) -> Self {
    //    match N {
    //        0 => bessel_j0::<Self, P>(self),
    //        _ => todo!(),
    //    }
    //}

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
        //
        // W has a square-root singularity at x = -1/e (double root of w*e^w - x at w = -1),
        // so Halley degenerates to linear convergence without a sqrt-based initial guess.
        //
        // The raw Puiseux series is W ≈ -1 ± (p - p^2/3 + 11p^3/72) where p = sqrt(2(ex+1)),
        // with + for W_0 and - for W_{-1}. This converges well near -1/e but diverges further
        // out. We damp it with a denominator that grows with distance from -1/e:
        //
        //   w_branch = -1 ± p*(1 + p*(-1/3 + p*11/72)) / (1 + K*p_0*p)
        //
        // where p_0 = ex+1, and K = 1/(C * e^(3/2) * sqrt(2)) with C ≈ 1.2144578338 found by
        // minimizing the integrated backward error |w*e^w - x| over [-1/e, 0] in Desmos.
        // The denominator arises from (x + 1/e)^1.5 / C = (p_0/e)^1.5 / C = p_0*p / (C*e^(3/2)*sqrt(2)).

        let p0 = x.mul_adde(Self::E, Self::ONE); // ex + 1
        let p = (p0 + p0).sqrt(); // sqrt(2(ex+1))

        // p*(1 + p*(-1/3 + p*11/72))
        let puiseux_numer = p * p.mul_adde(
            p.mul_adde(
                thermite::const_splat!(f32: 11.0 / 72.0),
                thermite::const_splat!(f32: -1.0 / 3.0),
            ),
            Self::ONE,
        );

        // 1 + K*p_0*p
        let puiseux_denom = p0.mul_adde(p * thermite::const_splat!(f32: 0.12991546098765432), Self::ONE);

        let puiseux = puiseux_numer / puiseux_denom;

        // W_0 branch: -1 + series, W_{-1} branch: -1 - series
        let w0_branch = puiseux + Self::NEG_ONE;
        let wm1_branch = Self::NEG_ONE - puiseux;

        // W_0 middle region: ex/(2+ex)
        let ex = x * Self::E;
        let w0_mid = ex / (Self::TWO + ex);

        // Shared ln for asymptotic regions
        let lnx = x.abs().ln_p::<Approx<P>>();

        // W_0 asymptotic (x > e): L_1 - L_2 + L_2/L_1 where L_1 = ln(x), L_2 = ln(L_1).
        // The L_2/L_1 correction is 0 at x = e (since L_2 = ln(1) = 0), so it doesn't
        // overshoot near the transition, but closes the gap at large x.
        let l2 = lnx.ln_p::<Approx<P>>();

        let w0_asymptotic = if const { P::POLICY.precision.le(PrecisionPolicy::Average) && V::HAS_APPROX_RCP } {
            l2.mul_adde(lnx.reciprocal_p::<Approx<P>>(), lnx - l2)
        } else {
            (lnx - l2) + (l2 / lnx)
        };

        // W_{-1} asymptotic (x near 0^-): L_1 - L_2 where L_1 = ln(-x), L_2 = ln(-L_1)
        // lnx = ln(|x|) = ln(-x) since x < 0; this is negative for small |x|.
        // -lnx is positive, so (-lnx).ln() = ln(-ln(-x)) = L_2.
        let wm1_asymptotic = lnx - (-lnx).ln_p::<Approx<P>>();

        // Select initial guesses
        let near_branch = x.cmp_lt(thermite::const_splat!(f32: -0.1));
        let large = x.cmp_gt(Self::E);

        let mut w0 = near_branch.select(w0_branch, large.select(w0_asymptotic, w0_mid));

        let near_branch_m1 = x.cmp_lt(thermite::const_splat!(f32: -0.25));
        let mut wm1 = near_branch_m1.select(wm1_branch, wm1_asymptotic);

        // --- Interleaved Halley iterations ---
        #[inline(always)]
        fn halley_step<P: Policy, W>(w: W, x: W) -> W
        where
            W: FloatVectorWithBits<Element = f32> + SpecializedTranscendentalMath<f32>,
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
        let num_iters = if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } { 2 } else { 1 };

        // warmup iteration with the looser precision to get close enough
        // for the main iterations to converge in the target precision
        w0 = halley_step::<Approx<P>, Self>(w0, x);
        wm1 = halley_step::<Approx<P>, Self>(wm1, x);

        let mut _iter = 0usize;
        while _iter < num_iters {
            _iter += 1;
            // don't need to check overflow within these since it should be well-defined for
            // all intermediate values, and the final check will catch any issues.
            w0 = halley_step::<CheckOverflow<P, false>, Self>(w0, x);
            wm1 = halley_step::<CheckOverflow<P, false>, Self>(wm1, x);
        }

        // --- Edge cases ---
        if const { P::POLICY.precision.ge(PrecisionPolicy::Average) } {
            let x_is_zero = x.is_zero();

            // At x = -1/e, both W_0 and W_{-1} = -1
            w0 = x.cmp_eq(Self::FRAC_NEG_1_E).select(Self::NEG_ONE, w0);
            // Honestly the approximation handles W_0(0) = 0 pretty well,
            // but just in case, explicitly set it to the correct value.
            w0 = w0.nz(x_is_zero); // W_0(0) = 0

            wm1 = x.cmp_eq(Self::FRAC_NEG_1_E).select(Self::NEG_ONE, wm1);
            wm1 = x_is_zero.select(Self::NEG_INFINITY, wm1); // W_{-1}(0) = -inf
        }

        if const { matches!(P::POLICY.denormal_behavior, DenormalBehavior::Preserve) } {
            // for subnormal inputs, W_0(x) ≈ x
            w0 = x.is_subnormal().select(x, w0);

            // NOTE: Somehow wm1 handles denormals fine on its own,
            // at least to the accuracy of the reference crate,
            // so we don't actually need this.
            //
            // wm1 = is_subnormal.select(wm1_asymptotic, wm1);
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
        erf_f_internal::<Self, P, false, false>(self, &mut V::EMPTY)
    }

    #[inline(always)]
    #[allow(const_item_mutation)]
    fn erfc<P: Policy>(self) -> Self {
        erf_f_internal::<Self, P, true, false>(self, &mut V::EMPTY)
    }

    #[inline(always)]
    fn logistic_sigmoid<P: Policy>(self) -> Self {
        if const { P::POLICY.precision.gt(PrecisionPolicy::Average) } {
            let is_pos = self.is_positive();
            let x = self.neg_c(is_pos); // conditionally negate if positive
            let e = x.exp_p::<P>();

            let n = is_pos.select(Self::ONE, e);
            let d = Self::ONE + e;

            return n / d;
        }

        (Self::ONE + (-self).exp_p::<P>()).reciprocal_p::<ExtraPrecision<P>>()
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

        // 36 is the largest integer whose factorial is finite in f32.
        generic::gamma::tgamma_impl::<P, _, _, _>(z, &crate::tables::LANCZOS_F32, 36.0, crate::tables::LN_MAX_F32)
    }

    #[inline(always)]
    fn trigamma<P: Policy>(self) -> Self {
        generic::trigamma::trigamma_impl::<P, _, _>(self, &crate::tables::TRIGAMMA_F32)
    }

    #[inline(always)]
    fn digamma<P: Policy>(self) -> Self {
        generic::digamma::digamma_impl::<P, _, _, _, _, _, _>(self, &crate::tables::DIGAMMA_F32)
    }

    #[inline(always)]
    fn beta<P: Policy>(a: Self, b: Self) -> Self {
        generic::gamma::beta_impl::<P, _, _, _>(a, b, &crate::tables::LANCZOS_F32)
    }

    #[inline(always)]
    fn expint<P: Policy, const N: usize>(self) -> Self {
        generic::expint::expint_double::<P, f32, Self, N>(self)
    }

    #[inline(always)]
    fn expint_primal<P: Policy, const N: usize>(self) -> (Self, Self) {
        generic::expint::expint_double_primal::<P, f32, Self, N>(self)
    }
}

// TEMP(bessel_j): dead while `bessel_j` is off the public trait. Kept, not deleted,
// because it is the working f32 `J_0` kernel and comes back with the rest of the
// family. Re-enable it together with the other TEMP(bessel_j) markers.
#[allow(dead_code)]
#[inline(always)]
fn bessel_j0_pqzero<V, P: Policy>(x: V, ix: V::Bits) -> (V, V)
where
    V: FloatVectorWithBits<Element = f32> + SpecializedSpecialMath<f32>,
{
    /* The asymptotic expansions of pzero is
     *      1 - 9/128 s^2 + 11025/98304 s^4 - ...,  where s = 1/x.
     * For x >= 2, We approximate pzero by
     *      pzero(x) = 1 + (R/S)
     * where  R = pR0 + pR1*s^2 + pR2*s^4 + ... + pR5*s^10
     *        S = 1 + pS0*s^2 + ... + pS4*s^10
     * and
     *      | pzero(x)-1-R/S | <= 2  ** ( -60.26)
     */
    const PR8: [f32; 6] = [
        /* for x in [inf, 8]=1/[0,0.125] */
        -5.2530439453e+03, /* 0xc5a4285a */
        -2.4852163086e+03, /* 0xc51b5376 */
        -2.5706311035e+02, /* 0xc3808814 */
        -8.0816707611e+00, /* 0xc1014e86 */
        -7.0312500000e-02, /* 0xbd900000 */
        0.0000000000e+00,  /* 0x00000000 */
    ];
    const PS8: [f32; 5] = [
        4.7627726562e+04, /* 0x473a0bba */
        1.1675296875e+05, /* 0x47e4087c */
        4.0597855469e+04, /* 0x471e95db */
        3.8337448730e+03, /* 0x456f9beb */
        1.1653436279e+02, /* 0x42e91198 */
    ];
    const PR5: [f32; 6] = [
        /* for x in [8,4.5454]=1/[0.125,0.22001] */
        -3.4643338013e+02, /* 0xc3ad3779 */
        -3.3123129272e+02, /* 0xc3a59d9b */
        -6.7674766541e+01, /* 0xc287597b */
        -4.1596107483e+00, /* 0xc0851b88 */
        -7.0312492549e-02, /* 0xbd8fffff */
        -1.1412546255e-11, /* 0xad48c58a */
    ];
    const PS5: [f32; 5] = [
        2.4060581055e+03, /* 0x451660ee */
        9.6254453125e+03, /* 0x461665c8 */
        5.9789707031e+03, /* 0x45bad7c4 */
        1.0512523193e+03, /* 0x44836813 */
        6.0753936768e+01, /* 0x42730408 */
    ];

    const PR3: [f32; 6] = [
        /* for x in [4.547,2.8571]=1/[0.2199,0.35001] */
        -3.1447946548e+01, /* 0xc1fb9565 */
        -5.8079170227e+01, /* 0xc2685112 */
        -2.1965976715e+01, /* 0xc1afba52 */
        -2.4090321064e+00, /* 0xc01a2d95 */
        -7.0311963558e-02, /* 0xbd8fffb8 */
        -2.5470459075e-09, /* 0xb12f081b */
    ];
    const PS3: [f32; 5] = [
        1.7358093262e+02, /* 0x432d94b8 */
        1.1279968262e+03, /* 0x448cffe6 */
        1.1936077881e+03, /* 0x44953373 */
        3.6151397705e+02, /* 0x43b4c1ca */
        3.5856033325e+01, /* 0x420f6c94 */
    ];

    const PR2: [f32; 6] = [
        /* for x in [2.8570,2]=1/[0.3499,0.5] */
        -3.2336456776e+00, /* 0xc04ef40d */
        -1.1193166733e+01, /* 0xc1331736 */
        -7.6356959343e+00, /* 0xc0f4579f */
        -1.4507384300e+00, /* 0xbfb9b1cc */
        -7.0303097367e-02, /* 0xbd8ffb12 */
        -8.8753431271e-08, /* 0xb3be98b7 */
    ];
    const PS2: [f32; 5] = [
        1.4657617569e+01, /* 0x416a859a */
        1.5387539673e+02, /* 0x4319e01a */
        2.7047027588e+02, /* 0x43873c32 */
        1.3620678711e+02, /* 0x430834f0 */
        2.2220300674e+01, /* 0x41b1c32d */
    ];

    /* For x >= 8, the asymptotic expansions of qzero is
     *      -1/8 s + 75/1024 s^3 - ..., where s = 1/x.
     * We approximate pzero by
     *      qzero(x) = s*(-1.25 + (R/S))
     * where  R = qR0 + qR1*s^2 + qR2*s^4 + ... + qR5*s^10
     *        S = 1 + qS0*s^2 + ... + qS5*s^12
     * and
     *      | qzero(x)/s +1.25-R/S | <= 2  ** ( -61.22)
     */
    const QR8: [f32; 6] = [
        /* for x in [inf, 8]=1/[0,0.125] */
        3.7014625000e+04, /* 0x471096a0 */
        8.8591972656e+03, /* 0x460a6cca */
        5.5767340088e+02, /* 0x440b6b19 */
        1.1768206596e+01, /* 0x413c4a93 */
        7.3242187500e-02, /* 0x3d960000 */
        0.0000000000e+00, /* 0x00000000 */
    ];
    const QS8: [f32; 6] = [
        -3.4389928125e+05, /* 0xc8a7eb69 */
        8.4050156250e+05,  /* 0x494d3359 */
        8.0330925000e+05,  /* 0x49441ed4 */
        1.4253829688e+05,  /* 0x480b3293 */
        8.0983447266e+03,  /* 0x45fd12c2 */
        1.6377603149e+02,  /* 0x4323c6aa */
    ];

    const QR5: [f32; 6] = [
        /* for x in [8,4.5454]=1/[0.125,0.22001] */
        1.9899779053e+03, /* 0x44f8bf4b */
        1.0272437744e+03, /* 0x448067cd */
        1.3511157227e+02, /* 0x43071c90 */
        5.8356351852e+00, /* 0x40babd86 */
        7.3242180049e-02, /* 0x3d95ffff */
        1.8408595828e-11, /* 0x2da1ec79 */
    ];
    const QS5: [f32; 6] = [
        -5.3543427734e+03, /* 0xc5a752be */
        3.5976753906e+04,  /* 0x470c88c1 */
        5.6751113281e+04,  /* 0x475daf1d */
        1.8847289062e+04,  /* 0x46933e94 */
        2.0778142090e+03,  /* 0x4501dd07 */
        8.2776611328e+01,  /* 0x42a58da0 */
    ];

    const QR3: [f32; 6] = [
        /* for x in [4.547,2.8571]=1/[0.2199,0.35001] */
        1.6673394775e+02, /* 0x4326bbe4 */
        1.7080809021e+02, /* 0x432acedf */
        4.2621845245e+01, /* 0x422a7cc5 */
        3.3442313671e+00, /* 0x405607e3 */
        7.3241114616e-02, /* 0x3d95ff70 */
        4.3774099900e-09, /* 0x3196681b */
    ];
    const QS3: [f32; 6] = [
        -1.4924745178e+02, /* 0xc3153f59 */
        2.5163337402e+03,  /* 0x451d4557 */
        6.4604252930e+03,  /* 0x45c9e367 */
        3.7041481934e+03,  /* 0x4567825f */
        7.0968920898e+02,  /* 0x44316c1c */
        4.8758872986e+01,  /* 0x42430916 */
    ];

    const QR2: [f32; 6] = [
        /* for x in [2.8570,2]=1/[0.3499,0.5] */
        1.6252708435e+01, /* 0x4182058c */
        3.1666231155e+01, /* 0x41fd5471 */
        1.4495602608e+01, /* 0x4167edfd */
        1.9981917143e+00, /* 0x3fffc4bf */
        7.3223426938e-02, /* 0x3d95f62a */
        1.5044444979e-07, /* 0x342189db */
    ];
    const QS2: [f32; 6] = [
        -5.3109550476e+00, /* 0xc0a9f358 */
        2.1266638184e+02,  /* 0x4354aa98 */
        8.8293585205e+02,  /* 0x445cbbe5 */
        8.4478375244e+02,  /* 0x44533229 */
        2.6934811401e+02,  /* 0x4386ac8f */
        3.0365585327e+01,  /* 0x41f2ecb8 */
    ];

    let z = x.reciprocal_p::<P>();
    let z2 = z * z;

    let m8 = ix.cmp_ge(thermite::const_splat!(u32: 0x41000000)); // |x| >= 8.0
    let m5 = ix.cmp_ge(thermite::const_splat!(u32: 0x409173eb)); // |x| >= 4.5454
    let m3 = ix.cmp_ge(thermite::const_splat!(u32: 0x4036d917)); // |x| >= 2.8571

    // Evaluate numerators and denominators for all 4 regions independently,
    // then select before dividing once.
    let pn8 = z2.poly_rev_p::<P, _>(&PR8);
    let pn5 = z2.poly_rev_p::<P, _>(&PR5);
    let pn3 = z2.poly_rev_p::<P, _>(&PR3);
    let pn2 = z2.poly_rev_p::<P, _>(&PR2);

    let pd8 = z2.poly_rev_p::<P, _>(&PS8);
    let pd5 = z2.poly_rev_p::<P, _>(&PS5);
    let pd3 = z2.poly_rev_p::<P, _>(&PS3);
    let pd2 = z2.poly_rev_p::<P, _>(&PS2);

    let pn = m3.select(m5.select(m8.select(pn8, pn5), pn3), pn2);
    let pd = m3.select(m5.select(m8.select(pd8, pd5), pd3), pd2);

    let qn8 = z2.poly_rev_p::<P, _>(&QR8);
    let qn5 = z2.poly_rev_p::<P, _>(&QR5);
    let qn3 = z2.poly_rev_p::<P, _>(&QR3);
    let qn2 = z2.poly_rev_p::<P, _>(&QR2);

    let qd8 = z2.poly_rev_p::<P, _>(&QS8);
    let qd5 = z2.poly_rev_p::<P, _>(&QS5);
    let qd3 = z2.poly_rev_p::<P, _>(&QS3);
    let qd2 = z2.poly_rev_p::<P, _>(&QS2);

    let qn = m3.select(m5.select(m8.select(qn8, qn5), qn3), qn2);
    let qd = m3.select(m5.select(m8.select(qd8, qd5), qd3), qd2);

    let pzero = V::ONE + pn / pd.mul_adde(z2, V::ONE);
    let mut qzero = qn / qd.mul_adde(z2, V::ONE);

    let neg_eighth: V = thermite::const_splat!(f32: -0.125);

    if const { V::HAS_TRUE_FMA } {
        // z*-1/8 can be computed earlier,
        // so despite this having the same number of
        // instructions as the non-FMA version, it will
        // be slightly faster
        qzero = qzero.mul_add(z, z * neg_eighth);
    } else {
        qzero = (qzero + neg_eighth) * z;
    }

    (pzero, qzero)
}

// TEMP(bessel_j): see above.
#[cfg(any())]
#[inline(always)]
fn bessel_j0<V, P: Policy>(x: V) -> V
where
    V: FloatVectorWithBits<Element = f32> + SpecializedSpecialMath<f32>,
{
    let ax = x.abs().flush_denormals_p::<P>();
    let ix: V::Bits = ax.into_bits();
    let large = ix.cmp_ge(thermite::const_splat!(u32: 0x40000000)); // |x| >= 2.0

    // ========================================================
    // Small-x path: |x| < 2
    // J_0(x) ≈ (1+x/2)(1-x/2) + z*(R(z)/S(z)),  z = x^2
    // The (1+x/2)(1-x/2) form avoids cancellation vs 1-x^2/4.
    // ========================================================
    let z = x * x;

    /* R0/S0 on [0, 2.00] */
    let r = z * z.poly_rev_p::<P, _>(&[
        -4.6183270541e-09, /* 0xb19eaf3c */
        1.8295404516e-06,  /* 0x35f58e88 */
        -1.8997929874e-04, /* 0xb947352e */
        1.5625000000e-02,  /* 0x3c800000 */
    ]);

    let s = z.poly_rev_p::<P, _>(&[
        1.1661400734e-09, /* 0x30a045e8 */
        5.1354652442e-07, /* 0x3509daa6 */
        1.1692678527e-04, /* 0x38f53697 */
        1.5619102865e-02, /* 0x3c7fe744 */
    ]);

    let s = s.mul_adde(z, V::ONE);

    let mut y = ax
        .mul_adde(V::HALF, V::ONE)
        .mul_adde(ax.nmul_adde(V::HALF, V::ONE), z * (r / s));

    // ========================================================
    // Large-x path: |x| >= 2
    // J_0(x) = FRAC_1_SQRT_PI * (P(x)*cc - Q(x)*ss) / sqrt(x)
    //
    // cc and ss encode cos(x-π/4) and sin(x-π/4) via a
    // numerical conditioning trick to avoid cancellation.
    // ========================================================

    if large.any() {
        let (sinx, cosx) = ax.sin_cos_p::<P>();

        // -cos(2x): fresh trig call at doubled argument for Best+ precision
        // (avoids cancellation in 1-2cos^2x near x ≈ kπ/4);
        // otherwise 1-2cos^2x, which is exact at the cancellation point
        // and only loses bits near - but not at - those values.
        let neg_cos2x = if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            -(ax + ax).cos_p::<P>()
        } else {
            (cosx * cosx).nmul_adde(V::TWO, V::ONE)
        };

        // cc = sin(x) + cos(x),  ss = sin(x) - cos(x)
        // Identity: cc * ss = sin^2x - cos^2x = -cos(2x)
        // Whichever of |cc|, |ss| is smaller gets recomputed
        // as -cos(2x) / (the larger one) for better precision.
        let cc_raw = sinx + cosx;
        let ss_raw = sinx - cosx;

        let fix_cc = (sinx * cosx).is_negative();
        let ratio = neg_cos2x / fix_cc.select(ss_raw, cc_raw);
        let cc = fix_cc.select(ratio, cc_raw);
        let ss = fix_cc.select(ss_raw, ratio);

        // Envelope polynomials (combined to share masks and 1/x^2)
        let (pz, qz) = bessel_j0_pqzero::<V, P>(ax, ix);

        let yl = V::FRAC_1_SQRT_PI * (pz * cc - qz * ss) / ax.sqrt();

        y = large.select(yl, y);

        if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            let very_large = ix.cmp_ge(thermite::const_splat!(u32: 0x7f800000));

            y = very_large.select(ax.square().reciprocal_p::<P>(), y);
        }
    }

    y
}

impl<V: FloatVectorWithBits<Element = f32>> SpecializedRealSpecialMath<f32> for V
where
    V: TranscendentalMathWithPolicy<Element = f32>,
    V: SpecializedTranscendentalMath<f32>,
{
    #[inline(always)]
    fn erfinv<P: Policy>(self) -> Self {
        // (-1, 1) range
        let x = self.flush_denormals_p::<P>().clamp(
            thermite::const_splat!(f32: -0.99999),
            thermite::const_splat!(f32: 0.99999),
        );

        let w = -x.nmul_adde(x, V::ONE).ln_p::<P>();

        let ge5 = w.cmp_ge(thermite::const_splat!(f32: 5.0));

        let w0 = w - thermite::const_splat!(f32: 2.5);
        let mut p0 = w0.poly_rev_p::<P, _>(&[
            2.81022636e-08,
            3.43273939e-07,
            -3.5233877e-06,
            -4.39150654e-06,
            0.00021858087,
            -0.00125372503,
            -0.00417768164,
            0.246640727,
            1.50140941,
        ]);

        if const { P::POLICY.avoid_branching } || thermite::unlikely(ge5.any()) {
            let w1 = w.sqrt() - thermite::const_splat!(f32: 3.0);
            let p1 = w1.poly_rev_p::<P, _>(&[
                -0.000200214257,
                0.000100950558,
                0.00134934322,
                -0.00367342844,
                0.00573950773,
                -0.0076224613,
                0.00943887047,
                1.00167406,
                2.83297682,
            ]);

            p0 = ge5.select(p1, p0);
        }

        p0 * x
    }

    /// Uses the algorithm from Peter John Acklam, sourced from here:
    /// <https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/>
    fn probit<P: Policy>(self) -> Self {
        const A: [f32; 6] = [
            2.506628277459239e+00,
            -3.066479806614716e+01,
            1.383577518672690e+02,
            -2.759285104469687e+02,
            2.209460984245205e+02,
            -3.969683028665376e+01,
        ];

        const B: [f32; 6] = [
            1.0,
            -1.328068155288572e+01,
            6.680131188771972e+01,
            -1.556989798598866e+02,
            1.615858368580409e+02,
            -5.447609879822406e+01,
        ];

        const C: [f32; 6] = [
            2.938163982698783e+00,
            4.374664141464968e+00,
            -2.549732539343734e+00,
            -2.400758277161838e+00,
            -3.223964580411365e-01,
            -7.784894002430293e-03,
        ];

        const D: [f32; 5] = [
            1.0,
            3.754408661907416e+00,
            2.445134137142996e+00,
            3.224671290700398e-01,
            7.784695709041462e-03,
        ];

        // f32 is at its precision limit without refinement (REFINE = false).
        generic::probit::probit_acklam::<P, _, _, false>(self, &A, &B, &C, &D)
    }

    #[inline(always)]
    fn lgamma_r<P: Policy>(self) -> (Self, Self) {
        let z = self.flush_denormals_p::<P>();
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
            if const { P::POLICY.avoid_branching } || thermite::unlikely(reflect.any()) {
                let pix = (z * Self::PI).sin_p::<P>();

                signum |= reflect.select(pix.signed_zero(), signum);

                e = reflect.select(pix.abs() / x, x);
                y = reflect.select(Self::LN_PI - y, y);
            }

            y -= e.ln_p::<P>();

            return (y, signum);
        }

        generic::gamma::lgamma_r_impl::<P, _, _, _>(z, &crate::tables::LANCZOS_F32)
    }

    #[inline(always)]
    fn gelu<P: Policy>(self, alpha: Self) -> Self {
        let x = self;

        let alpha_x = alpha * x;

        // GELU(x) = 0.5 * x * (1 + erf(ax / sqrt(2)))
        // O = false: skip the exp(-ax^2) byproduct that only the derivative needs.
        let mut unused = Self::EMPTY;
        let erf = erf_f_internal::<V, P, false, false>(alpha_x * Self::FRAC_1_SQRT_2, &mut unused);

        if V::HAS_TRUE_FMA {
            let half_x = x * Self::HALF;
            half_x.mul_add(erf, half_x) // fma(0.5x, erf, 0.5x), one rounding
        } else {
            erf.mul_adde(Self::HALF, Self::HALF) * x
        }
    }
}

impl<V: FloatVectorWithBits<Element = f32>> SpecializedRealPrimalMath<f32> for V {
    #[inline(always)]
    fn gelu_d<P: Policy>(self, alpha: Self) -> (Self, Self) {
        let x = self;

        let alpha_x = alpha * x;

        // GELU(x) = 0.5 * x * (1 + erf(ax / sqrt(2)))
        let mut exp_neg_ax2 = Self::EMPTY;
        let erf = erf_f_internal::<V, P, false, true>(alpha_x * Self::FRAC_1_SQRT_2, &mut exp_neg_ax2);

        let y;
        let dy;

        let half_erf = erf.mul_adde(Self::HALF, Self::HALF);

        let alpha_x_scaled = x.scale(FloatConsts::FRAC_1_SQRT_TAU);

        if V::HAS_TRUE_FMA {
            let half_x = x * Self::HALF;
            y = half_x.mul_add(erf, half_x); // fma(0.5x, erf, 0.5x), one rounding
            dy = alpha_x_scaled.mul_add(exp_neg_ax2, half_erf);
        } else {
            y = half_erf * x;
            dy = half_erf + alpha_x_scaled * exp_neg_ax2;
        }

        (y, dy)
    }
}

#[allow(clippy::approx_constant)]
#[inline(always)]
fn erf_f_internal<V: FloatVectorWithBits<Element = f32>, P: Policy, const C: bool, const O: bool>(
    x0: V,
    out_exp_neg_x2: &mut V,
) -> V {
    // Extract the sign bit once. abs(x0) = x0 ^ sign, and sign is reused
    // for the final operation in every branch, avoiding a redundant bitand.
    let sign = x0.signed_zero();
    let x = (x0 ^ sign).flush_denormals_p::<P>();

    // NOTE: For GPUs, exp is usually free, so these approximations are actually more expensive
    // than just using exp, but for CPUs they can be much faster, and the precision is still decent for many use cases.
    if const {
        matches!(P::POLICY.precision, PrecisionPolicy::Worst | PrecisionPolicy::Medium if !V::NATIVE_CAP.has(NativeCapability::EXP))
    } {
        // the polynomials below are sensitive to large inputs, so we need to clamp x to avoid exploding into inf/nan,
        // and erf(x) is saturating to 1.0 around x=3.81, so 4.5 is a safe clamping point that won't cause significant precision
        // loss for large inputs, but will prevent overflow in the polynomial evaluation.
        let x = x.min(thermite::const_splat!(f32: 4.5));

        // Both use erf(x) ≈ 1 - 1/t^n for a polynomial t; only the poly and
        // exponent differ. Worst: A&S degree-4, t^4.  Medium: A&S 7.1.27 degree-6, t^16 (3e-7).
        let tn = if const { matches!(P::POLICY.precision, PrecisionPolicy::Worst) } {
            let t = x.poly_rev_p::<P, _>(&[0.078108, 0.000972, 0.230389, 0.278393, 1.0]);

            t.powi_p::<P>(4)
        } else {
            let t = x.poly_rev_p::<P, _>(&[
                0.0000430638,
                0.0002765672,
                0.0001520143,
                0.0092705272,
                0.0422820123,
                0.0705230784,
                1.0,
            ]);

            t.powi_p::<P>(16)
        };

        if const { O } {
            // We need a relatively accurate exp(-x^2) for GELU derivative, so opt for medium precision even in worst case,
            // which is still much cheaper than a full exp.
            *out_exp_neg_x2 = (-x * x).exp_p::<MediumPrecision<CheckOverflow<P, false>>>();
        }

        match const { (C, V::HAS_APPROX_RCP) } {
            (false, true) => {
                let y = tn.rcp();
                y.nmul_adde(tn.nmul_adde(y, V::TWO), V::ONE) ^ sign
            }
            (false, false) => (V::ONE - tn.reciprocal_p::<ExtraPrecision<P>>()) ^ sign,
            (true, true) => {
                let y = tn.rcp();
                let k = tn.nmul_adde(y, V::TWO);

                if V::HAS_TRUE_FMA {
                    sign.select_negative(y.nmul_add(k, V::TWO), y * k)
                } else {
                    let erfc_pos = y * k;
                    sign.select_negative(V::TWO - erfc_pos, erfc_pos)
                }
            }
            (true, false) => {
                let y = tn.reciprocal_p::<ExtraPrecision<P>>();
                sign.select_negative(V::TWO - y, y)
            }
        }
    }
    // higher precision policies or GPU with native exp support.
    else {
        // NOTE: x does not need to be clamped here, everything behaves well even for large inputs.

        // if ignoring denormals, just multiple x0 by itself to save like one cycle,
        // instead of waiting on abs(), otherwise use the denormal-flushed x value
        let x2 = if const { matches!(P::POLICY.denormal_behavior, DenormalBehavior::Ignore) } {
            x0 * x0
        } else {
            x * x
        };

        let exp_neg_x2 = (-x2).exp_p::<P>();

        // Improved A&S method from Wikipedia, max error ~2e-9
        let p1: V = thermite::const_splat!(f32: 0.406742016006509);
        let p2: V = thermite::const_splat!(f32: 0.0072279182302319);

        let t = x.mul_adde(x.mul_adde(p2, p1), V::ONE).reciprocal_p::<P>();

        let m = t.poly_rev_p::<P, _>(&[
            0.0382613542530727,
            -0.393127715207728,
            1.20644903073232,
            -1.11694155120396,
            1.08680830347054,
            -0.138329314150635,
            0.316879890481381, // A1
        ]);

        if const { O } {
            *out_exp_neg_x2 = exp_neg_x2;
        }

        // NOTE: We multiple e by t here, instead of
        // t * t.poly, as this noticeably
        // improve precision at zero cost.
        let e = exp_neg_x2 * t;

        if const { C } {
            if const { V::HAS_TRUE_FMA && P::POLICY.precision.lt(PrecisionPolicy::Average) } {
                return sign.select_negative(e.nmul_add(m, V::TWO), e * m);
            }

            let mut y = e * m;

            let is_big = x.cmp_gt(V::ONE);

            if const { P::POLICY.precision.ge(PrecisionPolicy::Average) }
                && (const { P::POLICY.avoid_branching } || is_big.any())
            {
                let s = x.reciprocal_p::<P>();

                let big_y = if const { P::POLICY.precision.ge(PrecisionPolicy::Reference) } {
                    // slow reference code from libm, matches nearly exactly to libm itself.
                    let r = s.poly_rev_p::<P, _>(&[
                        -4.8351919556e+02,
                        -1.0250950928e+03,
                        -6.3756646729e+02,
                        -1.6063638306e+02,
                        -1.7757955551e+01,
                        -7.9928326607e-01,
                        -9.8649431020e-03,
                    ]);

                    let b = s.poly_rev_p::<P, _>(&[
                        -2.2440952301e+01,
                        4.7452853394e+02,
                        2.5530502930e+03,
                        3.1998581543e+03,
                        1.5367296143e+03,
                        3.2579251099e+02,
                        3.0338060379e+01,
                        1.0,
                    ]);

                    let z: V = {
                        // Bit-split: zero low 13 mantissa bits so z*z is exact in f32.
                        let mut ix: V::Bits = x.into_bits();
                        ix &= thermite::const_splat!(u32: 0xffffe000);
                        ix.into_bits()
                    };

                    let a = (-z * z - thermite::const_splat!(f32: 0.5625)).exp_p::<CheckOverflow<P, false>>();
                    let b = ((z - x) * (z + x) + r / b).exp_p::<CheckOverflow<P, false>>() / x;

                    a * b
                } else {
                    // fast minimax approximation with a 68 ULP max difference, avg 0.282 ULP
                    exp_neg_x2
                        * s.mul_adde(
                            thermite::const_splat!(f32: 9.0 / 4.0),
                            thermite::const_splat!(f32: -5.0 / 4.0),
                        )
                        .poly_rev_p::<P, _>(&[
                            -1.5849000192247331142425537109375e-5,
                            4.057946716784499585628509521484375e-5,
                            -2.17467240872792899608612060546875e-5,
                            -9.03195686987601220607757568359375e-5,
                            4.285395261831581592559814453125e-4,
                            -1.16943917237222194671630859375e-3,
                            1.68157299049198627471923828125e-3,
                            3.04660876281559467315673828125e-3,
                            -3.5686969757080078125e-2,
                            0.18081049621105194091796875,
                            0.278560101985931396484375,
                        ])
                };

                y = is_big.select(big_y, y);
            }

            sign.select_negative(V::TWO - y, y)
        } else {
            let mut y = e.nmul_adde(m, V::ONE);

            if const { P::POLICY.precision.ge(PrecisionPolicy::Average) } {
                let small = if const { P::POLICY.precision.le(PrecisionPolicy::Average) } {
                    // Taylor series for erf(x)/x, faster but slightly less accurate at points
                    x * x2.poly_rev_p::<P, _>(&[
                        0.00012055332981789664251,
                        -0.00085483270234508528325,
                        0.0052239776254421878421,
                        -0.026866170645131251759,
                        0.11283791670955125739,
                        -0.37612638903183752463,
                        1.1283791670955125739,
                    ])
                } else {
                    // Pade approximate for (Erf(x)-x)/x
                    let n = x2.poly_rev_p::<P, _>(&[
                        -2.3763017452e-05,
                        -5.7702702470e-03,
                        -2.8481749818e-02,
                        -3.2504209876e-01,
                        1.2837916613e-01,
                    ]);

                    let d = x2.poly_rev_p::<P, _>(&[
                        -3.9602282413e-06,
                        1.3249473704e-04,
                        5.0813062117e-03,
                        6.5022252500e-02,
                        3.9791721106e-01,
                        1.0,
                    ]);

                    x.mul_adde(n / d, x)
                };

                y = x.cmp_lt(V::ONE).select(small, y);
            }

            y | sign
        }
    }
}

/// Every default applies: `expint` on the real line is what they were written for.
impl<V: FloatVectorWithBits<Element = f32>> super::ExpIntDetails<f32, V> for V {}
