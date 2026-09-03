use thermite::{
    math::{
        TranscendentalMathWithPolicy,
        policy::{
            DenormalBehavior, PrecisionPolicy,
            policies::{CheckOverflow, ExtraPrecision, WorstPrecision},
        },
        specialized::SpecializedTranscendentalMath,
        specialized::reference::{is_reference, map1, map1x2},
    },
    prelude::*,
};

use crate::RealSpecialMathWithPolicy as _;

use super::*;

impl<V: FloatVectorWithBits<Element = f64>> SpecializedSpecialMath<f64> for V
where
    V: TranscendentalMathWithPolicy<Element = f64>,
    V: SpecializedTranscendentalMath<f64>,
    // Pins the projection so `V`'s real-special methods (whose impl requires
    // `Primal = V`) resolve. A type parameter's `Primal` will not normalize
    // through the blanket impl on its own.
    V: thermite::math::PrimalProjection<Primal = V>,
    V: thermite::math::RealMathWithPolicy<Element = f64>,
{
    #[inline(always)]
    fn zetac<P: Policy>(self) -> Self {
        generic::zeta::zeta_impl::<P, _, _, true>(self)
    }

    #[inline(always)]
    fn polylog<P: Policy>(self, order: crate::PolylogOrder<f64, i64>) -> Self {
        generic::polylog::polylog_impl::<P, f64, Self>(self, order)
    }

    #[inline(always)]
    fn zeta<P: Policy>(self) -> Self {
        generic::zeta::zeta_impl::<P, _, _, false>(self)
    }

    #[inline(always)]
    fn zeta_with_deriv<P: Policy, const ZETAC: bool>(self) -> (Self, Self) {
        generic::zeta::zeta_core::<P, _, _, ZETAC, true>(self)
    }

    #[inline(always)]
    fn bessel_i<P: Policy, const N: i32>(self) -> Self {
        bessel_i_dispatch::<P, Self, N, false>(self)
    }

    #[inline(always)]
    fn bessel_i_scaled<P: Policy, const N: i32>(self) -> Self {
        bessel_i_dispatch::<P, Self, N, true>(self)
    }

    #[inline(always)]
    fn bessel_k<P: Policy, const N: i32>(self) -> Self {
        bessel_k_dispatch::<P, Self, N, false>(self)
    }

    #[inline(always)]
    fn bessel_k_scaled<P: Policy, const N: i32>(self) -> Self {
        bessel_k_dispatch::<P, Self, N, true>(self)
    }

    #[inline(always)]
    fn bessel_j<P: Policy, const N: i32>(self) -> Self {
        use crate::tables::bessel::jy::{BESSEL_J0_F64, BESSEL_J1_F64};
        // `Reference` is contractually bit-identical to libm, lane by lane, and unlike most
        // of this crate, libm actually has these (the C/POSIX XSI set) at every order, so the
        // arm exists. `I`/`K` have no libm counterpart and therefore no reference arm.
        if const { is_reference::<P>() } {
            let v = if const { N == 0 } {
                map1(self, libm::j0)
            } else if const { N.unsigned_abs() == 1 } {
                map1(self, libm::j1)
            } else {
                map1(self, |v| libm::jn(N.abs(), v))
            };
            // A sign flip is exact, so reflecting libm's own value keeps the tier's
            // bit-identity promise rather than trading it for a second algorithm.
            return if const { bessel_reflect_negates(N) } { -v } else { v };
        }
        let v = if const { N == 0 } {
            generic::bessel::jy::bessel_j0_impl::<P, f64, _, _, _, _>(self, &BESSEL_J0_F64)
        } else if const { N.unsigned_abs() == 1 } {
            generic::bessel::jy::bessel_j1_impl::<P, f64, _, _, _, _>(self, &BESSEL_J1_F64)
        } else {
            generic::bessel::jy::bessel_jn_pair_impl::<P, f64, _, _, _, _, _, _, _, N>(
                self,
                &BESSEL_J0_F64,
                &BESSEL_J1_F64,
            )
            .1
        };
        // `J_{-n} = (-1)^n J_n`. Every arm above evaluated at `|N|`.
        if const { bessel_reflect_negates(N) } { -v } else { v }
    }

    #[inline(always)]
    fn bessel_y<P: Policy, const N: i32>(self) -> Self {
        use crate::tables::bessel::jy::{BESSEL_J0_F64, BESSEL_J1_F64, BESSEL_Y0_F64, BESSEL_Y1_F64};
        if const { is_reference::<P>() } {
            let v = if const { N == 0 } {
                map1(self, libm::y0)
            } else if const { N.unsigned_abs() == 1 } {
                map1(self, libm::y1)
            } else {
                map1(self, |v| libm::yn(N.abs(), v))
            };
            return if const { bessel_reflect_negates(N) } { -v } else { v };
        }
        let v = if const { N.unsigned_abs() >= 2 } {
            // Y is the dominant solution, so upward recurrence is stable and costs exactly
            // |N|-1 steps. No trip count question at all, unlike J.
            let y0 = generic::bessel::jy::bessel_y_impl::<P, f64, _, _, _, _, _, _, _, _, false>(
                self,
                &BESSEL_Y0_F64,
                &BESSEL_J0_F64,
            );
            let y1 = generic::bessel::jy::bessel_y_impl::<P, f64, _, _, _, _, _, _, _, _, true>(
                self,
                &BESSEL_Y1_F64,
                &BESSEL_J1_F64,
            );
            generic::bessel::jy::bessel_yn_recur::<f64, _, N>(self, y0, y1).1
        } else if const { N == 0 } {
            generic::bessel::jy::bessel_y_impl::<P, f64, _, _, _, _, _, _, _, _, false>(
                self,
                &BESSEL_Y0_F64,
                &BESSEL_J0_F64,
            )
        } else {
            generic::bessel::jy::bessel_y_impl::<P, f64, _, _, _, _, _, _, _, _, true>(
                self,
                &BESSEL_Y1_F64,
                &BESSEL_J1_F64,
            )
        };
        // `Y_{-n} = (-1)^n Y_n`, the same reflection `J` gets.
        if const { bessel_reflect_negates(N) } { -v } else { v }
    }

    #[inline(always)]
    fn bessel_i_with_deriv<P: Policy, const N: i32, const SCALED: bool>(self) -> (Self, Self) {
        bessel_i_deriv_dispatch::<P, Self, N, SCALED>(self)
    }

    #[inline(always)]
    fn bessel_k_with_deriv<P: Policy, const N: i32, const SCALED: bool>(self) -> (Self, Self) {
        bessel_k_deriv_dispatch::<P, Self, N, SCALED>(self)
    }

    #[inline(always)]
    fn bessel_iv<P: Policy, const SCALED: bool>(self, order: crate::BesselOrder<Self, Self::Signed>) -> Self {
        // Half-integer order is elementary: hyperbolic seeds and the same two recurrence
        // directions the integer kernel uses. See `generic::bessel_half`.
        let order = order.simplify();
        if let crate::BesselOrder::HalfInteger(k) = order {
            return generic::bessel::half::bessel_ik_half::<P, f64, _, SCALED>(
                Self::from_signed_integer(k) * Self::HALF,
                self,
                crate::tables::bessel::BESSEL_I0_F64.far_threshold,
            )
            .0;
        }
        // `Thirds` and `Real` take the table-free arms in `generic::bessel_ik_nu`.
        let Some(n) = order.as_integer() else {
            return generic::bessel::ik_real::bessel_ik_real::<P, f64, _, _, 25, 25, SCALED, true>(
                order.to_real(),
                self,
                &crate::tables::lgamma1p::LGAMMA1P_F64,
                crate::tables::bessel::BESSEL_I0_F64.far_threshold,
            )
            .0;
        };
        // `I_{-n} = I_n` for integer `n`, so only the magnitude matters and no sign is owed
        // afterwards. `J`/`Y` below are the ones that reflect.
        let nf = Self::from_signed_integer(n).abs();
        let v = generic::bessel::ik::bessel_iv_impl::<P, f64, _, _, _, _, SCALED>(
            self,
            nf,
            &crate::tables::bessel::BESSEL_I0_F64,
        );
        // Orders 0 and 1 have closed forms, and the ratio ladder is measurably worse at them:
        // it reaches order 1 as `I_0 * r_1`, paying the continued fraction for a value the
        // table gives directly. Measured 5.49 ULP against 3.04 before this select was added.
        let i1 =
            generic::bessel::ik::bessel_i1_impl::<P, _, _, _, _, SCALED>(self, &crate::tables::bessel::BESSEL_I1_F64);
        nf.cmp_le(Self::ONE).select(
            nf.cmp_le(Self::ZERO).select(
                generic::bessel::ik::bessel_i0_impl::<P, _, _, _, _, SCALED>(
                    self,
                    &crate::tables::bessel::BESSEL_I0_F64,
                ),
                i1,
            ),
            v,
        )
    }

    #[inline(always)]
    fn bessel_kv<P: Policy, const SCALED: bool>(self, order: crate::BesselOrder<Self, Self::Signed>) -> Self {
        // Half-integer order is elementary (see `generic::bessel_half`).
        let order = order.simplify();
        if let crate::BesselOrder::HalfInteger(k) = order {
            return generic::bessel::half::bessel_ik_half::<P, f64, _, SCALED>(
                Self::from_signed_integer(k) * Self::HALF,
                self,
                crate::tables::bessel::BESSEL_I0_F64.far_threshold,
            )
            .1;
        }
        // `Thirds` and `Real` take the table-free arms in `generic::bessel_ik_nu`.
        let Some(n) = order.as_integer() else {
            // `NEED_I = false`: this entry wants only `K`, which is the cheap half. Skipping
            // `I` skips the continued fraction and the asymptotic series both.
            return generic::bessel::ik_real::bessel_ik_real::<P, f64, _, _, 25, 25, SCALED, false>(
                order.to_real(),
                self,
                &crate::tables::lgamma1p::LGAMMA1P_F64,
                crate::tables::bessel::BESSEL_I0_F64.far_threshold,
            )
            .1;
        };
        // `K_{-n} = K_n`, as with `I`.
        let nf = Self::from_signed_integer(n).abs();
        generic::bessel::ik::bessel_kv_impl::<P, f64, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, SCALED>(
            self,
            nf,
            &crate::tables::bessel::BESSEL_K0_F64,
            &crate::tables::bessel::BESSEL_K1_F64,
            &crate::tables::bessel::BESSEL_I0_F64,
            &crate::tables::bessel::BESSEL_I1_F64,
        )
    }

    #[inline(always)]
    fn bessel_jv<P: Policy>(self, order: crate::BesselOrder<Self, Self::Signed>) -> Self {
        // Half-integer order is elementary (see `generic::bessel_half`). `simplify` has
        // already turned an even numerator into `Integer`, so anything still tagged
        // `HalfInteger` here is a genuine half-odd order.
        let order = order.simplify();
        if let crate::BesselOrder::HalfInteger(k) = order {
            return generic::bessel::half::bessel_jy_half::<P, f64, _>(Self::from_signed_integer(k) * Self::HALF, self)
                .0;
        }
        // `Thirds` and `Real` take the table-free arms in `generic::bessel_nu`, which cover
        // the whole axis at any real order. Thirds are not specialised beyond that, and
        // deliberately: see the module docs there.
        let Some(n) = order.as_integer() else {
            let nu = order.to_real();
            return generic::bessel::jy_real::bessel_jy_real::<P, f64, _, 29, 25, 25, 17, 1>(
                nu,
                self,
                Self::ZERO,
                &crate::tables::lgamma1p::LGAMMA1P_F64,
            )
            .0;
        };
        // The const form routes `Reference` to libm. So must this one, or the tier silently
        // stops meaning "bit-identical to libm" as soon as the order moves into a register.
        if const { is_reference::<P>() } {
            let mut out = self;
            let mut i = 0;
            while i < Self::LANES {
                // Reflected here rather than handed to libm signed, so the tier means the
                // same thing at negative order as the const form does.
                let k = n.extractv(i);
                let r = libm::jn(k.unsigned_abs() as i32, self.extractv(i));
                out = out.insertv(i, if k < 0 && k % 2 != 0 { -r } else { r });
                i += 1;
            }
            return out;
        }
        let (nf, flip) = bessel_reflect_v(Self::from_signed_integer(n));
        generic::bessel::jy::bessel_jv_impl::<P, f64, _, _, _, _, _, _, _>(
            self,
            nf,
            &crate::tables::bessel::jy::BESSEL_J0_F64,
            &crate::tables::bessel::jy::BESSEL_J1_F64,
        )
        .neg_c(flip)
    }

    #[inline(always)]
    fn bessel_yv<P: Policy>(self, order: crate::BesselOrder<Self, Self::Signed>) -> Self {
        // Half-integer order is elementary (see `generic::bessel_half`).
        let order = order.simplify();
        if let crate::BesselOrder::HalfInteger(k) = order {
            return generic::bessel::half::bessel_jy_half::<P, f64, _>(Self::from_signed_integer(k) * Self::HALF, self)
                .1;
        }
        let Some(n) = order.as_integer() else {
            let nu = order.to_real();
            return generic::bessel::jy_real::bessel_jy_real::<P, f64, _, 29, 25, 25, 17, 1>(
                nu,
                self,
                Self::ZERO,
                &crate::tables::lgamma1p::LGAMMA1P_F64,
            )
            .1;
        };
        if const { is_reference::<P>() } {
            let mut out = self;
            let mut i = 0;
            while i < Self::LANES {
                let k = n.extractv(i);
                let r = libm::yn(k.unsigned_abs() as i32, self.extractv(i));
                out = out.insertv(i, if k < 0 && k % 2 != 0 { -r } else { r });
                i += 1;
            }
            return out;
        }
        let (nf, flip) = bessel_reflect_v(Self::from_signed_integer(n));
        generic::bessel::jy::bessel_yv_impl::<P, f64, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _>(
            self,
            nf,
            &crate::tables::bessel::jy::BESSEL_Y0_F64,
            &crate::tables::bessel::jy::BESSEL_Y1_F64,
            &crate::tables::bessel::jy::BESSEL_J0_F64,
            &crate::tables::bessel::jy::BESSEL_J1_F64,
        )
        .neg_c(flip)
    }

    impl_sph_bessel_entries!(f64, crate::tables::bessel::BESSEL_I0_F64);

    impl_airy_entries!(
        f64,
        29,
        25,
        25,
        17,
        1,
        &crate::tables::lgamma1p::LGAMMA1P_F64,
        &crate::tables::bessel::airy::AIRY_ZERO_F64,
        crate::tables::bessel::BESSEL_I0_F64
    );

    #[inline(always)]
    fn bessel_j_with_deriv<P: Policy, const N: i32>(self) -> (Self, Self) {
        // Order N-1 comes from the recurrence, which walks through it either way: forward
        // passes it on the last step, downward keeps the shorter product.
        let (prev, v) = if const { N == 0 } {
            // J_{-1} = -J_1, so the identity still holds and the N/x term simply vanishes.
            (-Self::bessel_j::<P, 1>(self), Self::bessel_j::<P, 0>(self))
        } else if const { N.unsigned_abs() == 1 } {
            (Self::bessel_j::<P, 0>(self), Self::bessel_j::<P, 1>(self))
        } else {
            generic::bessel::jy::bessel_jn_pair_impl::<P, f64, _, _, _, _, _, _, _, N>(
                self,
                &crate::tables::bessel::jy::BESSEL_J0_F64,
                &crate::tables::bessel::jy::BESSEL_J1_F64,
            )
        };
        let d = if const { N == 0 } {
            prev
        } else {
            prev - v * (Self::splat(N.unsigned_abs() as f64) / self)
        };
        // The pair above is at `|N|`. Reflecting a negative order scales the function by a
        // constant `(-1)^n`, so differentiating both sides carries the identical sign.
        if const { bessel_reflect_negates(N) } {
            (-v, -d)
        } else {
            (v, d)
        }
    }

    #[inline(always)]
    fn bessel_y_with_deriv<P: Policy, const N: i32>(self) -> (Self, Self) {
        let (prev, v) = if const { N == 0 } {
            (-Self::bessel_y::<P, 1>(self), Self::bessel_y::<P, 0>(self))
        } else if const { N.unsigned_abs() == 1 } {
            (Self::bessel_y::<P, 0>(self), Self::bessel_y::<P, 1>(self))
        } else {
            let y0 = generic::bessel::jy::bessel_y_impl::<P, f64, _, _, _, _, _, _, _, _, false>(
                self,
                &crate::tables::bessel::jy::BESSEL_Y0_F64,
                &crate::tables::bessel::jy::BESSEL_J0_F64,
            );
            let y1 = generic::bessel::jy::bessel_y_impl::<P, f64, _, _, _, _, _, _, _, _, true>(
                self,
                &crate::tables::bessel::jy::BESSEL_Y1_F64,
                &crate::tables::bessel::jy::BESSEL_J1_F64,
            );
            generic::bessel::jy::bessel_yn_recur::<f64, _, N>(self, y0, y1)
        };
        let d = if const { N == 0 } {
            prev
        } else {
            prev - v * (Self::splat(N.unsigned_abs() as f64) / self)
        };
        // The pair above is at `|N|`. Reflecting a negative order scales the function by a
        // constant `(-1)^n`, so differentiating both sides carries the identical sign.
        if const { bessel_reflect_negates(N) } {
            (-v, -d)
        } else {
            (v, d)
        }
    }

    type ExpIntDetails = Self;
    const LAGUERRE_PRODUCT_SEED_CAP: i32 = 170;

    #[inline(always)]
    fn chebyshev_n<P: Policy, const K: usize, const N: usize>(self, coeffs: &[f64; N]) -> Self {
        // See the trait default: the kernel reads `N = 0` as "runtime length", so the
        // empty-series rejection belongs to the entry point.
        const {
            assert!(N >= 1, "chebyshev_n: N must be at least 1");
        }

        // Real vectors have copysign and a real nearest endpoint, so the Reinsch form is
        // available, but the kernel still gates it on the policy asking for `Best` or better.
        generic::chebyshev::chebyshev_series::<P, _, _, K, N, true>(self, coeffs)
    }

    #[inline(always)]
    fn chebyshev<P: Policy, const K: usize>(self, coeffs: &[f64]) -> Self {
        // Reinsch available here too, on the same terms. See `chebyshev_n`.
        generic::chebyshev::chebyshev_series::<P, _, _, K, 0, true>(self, coeffs)
    }

    // TEMP(bessel_j): disabled until orders beyond J_0 exist. See thermite-special/src/lib.rs.
    //fn bessel_j<P: Policy, const N: i32>(self) -> Self {
    //    todo!()
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
        if const { is_reference::<P>() } {
            return map1(self, libm::erf);
        }

        erf_d_internal::<Self, P, false, false>(self, &mut V::EMPTY)
    }

    #[inline(always)]
    #[allow(const_item_mutation)]
    fn erfc<P: Policy>(self) -> Self {
        if const { is_reference::<P>() } {
            return map1(self, libm::erfc);
        }

        erf_d_internal::<Self, P, true, false>(self, &mut V::EMPTY)
    }

    #[inline(always)]
    fn erfcx<P: Policy>(self) -> Self {
        // No libm counterpart at any tier. `erfcx` is not in the C library, and
        // `exp(x*x) * erfc(x)` is exactly the overflowing form this replaces.
        super::generic::erfcx::erfcx_internal::<Self, f64, P>(self)
    }

    #[inline(always)]
    fn lgamma<P: Policy>(self) -> Self {
        if const { is_reference::<P>() } {
            return map1(self, libm::lgamma);
        }

        Self::lgamma_r::<P>(self).0
    }

    #[inline(always)]
    fn tgamma<P: Policy>(self) -> Self {
        if const { is_reference::<P>() } {
            return map1(self, libm::tgamma);
        }

        let z = self;

        if const { P::POLICY.precision.lt(PrecisionPolicy::Average) } {
            // We have a good lgamma approximation, so use it for tgamma on lower precisions.
            let (lgamma, sign) = z.lgamma_r_p::<P>();

            // use min(P + 1, Average) precision here. We want decent precision,
            // but not more than average.
            return lgamma.exp_p::<ExtraPrecision<P>>() * sign;
        }

        // 172 is the largest integer whose factorial is finite in f64.
        generic::gamma::tgamma_impl::<P, _, _, _>(
            z,
            &crate::tables::gamma::LANCZOS_F64,
            172.0,
            crate::tables::gamma::LN_MAX_F64,
        )
    }

    #[inline(always)]
    fn trigamma<P: Policy>(self) -> Self {
        generic::trigamma::trigamma_impl::<P, _, _>(self, &crate::tables::gamma::TRIGAMMA_F64)
    }

    #[inline(always)]
    fn polygamma<P: Policy>(self, n: u32) -> Self {
        generic::polygamma::polygamma_impl::<P, _, _>(self, n)
    }

    #[inline(always)]
    fn digamma<P: Policy>(self) -> Self {
        generic::digamma::digamma_impl::<P, _, _, _, _, _, _>(self, &crate::tables::gamma::DIGAMMA_F64)
    }

    #[inline(always)]
    fn beta<P: Policy>(a: Self, b: Self) -> Self {
        generic::gamma::beta_impl::<P, _, _, _>(a, b, &crate::tables::gamma::LANCZOS_F64)
    }

    #[inline(always)]
    fn expint_n<P: Policy, const N: usize>(self) -> Self {
        generic::expint::expint_double_n::<P, f64, Self, N>(self)
    }

    #[inline(always)]
    fn expint_primal_n<P: Policy, const N: usize>(self) -> (Self, Self) {
        generic::expint::expint_double_primal_n::<P, f64, Self, N>(self)
    }

    #[inline(always)]
    fn phi_n<P: Policy, const N: usize>(self) -> Self {
        // Fixed series length for f64. See the f32 twin for the budget split.
        let terms = const {
            let needed =
                super::generic::phi::phi_series_terms(N, f64::EPSILON * P::POLICY.precision.tolerance() as f64 / 32.0);
            if needed < P::POLICY.max_iterations {
                needed
            } else {
                P::POLICY.max_iterations
            }
        };
        super::generic::phi::phi_internal_n::<Self, f64, P, N, false>(self, terms)
    }

    #[inline(always)]
    fn expint<P: Policy>(self, n: u32) -> Self {
        generic::expint::expint_double::<P, f64, Self>(self, n)
    }

    #[inline(always)]
    fn expint_primal<P: Policy>(self, n: u32) -> (Self, Self) {
        generic::expint::expint_double_primal::<P, f64, Self>(self, n)
    }

    #[inline(always)]
    fn phi<P: Policy>(self, n: u32) -> Self {
        // The const form's term counts, precomputed per policy. The search runs per call
        // only past the table. Same budget split as `phi_n`.
        const EPS_SCALE: f64 = 1.0 / 32.0;
        let table = const {
            super::generic::phi::phi_terms_table(
                f64::EPSILON * P::POLICY.precision.tolerance() as f64 * EPS_SCALE,
                P::POLICY.max_iterations,
            )
        };
        let terms = match table.get(n as usize) {
            Some(&t) => t,
            None => {
                let needed = super::generic::phi::phi_series_terms(
                    n as usize,
                    f64::EPSILON * P::POLICY.precision.tolerance() as f64 * EPS_SCALE,
                );
                if needed < P::POLICY.max_iterations {
                    needed
                } else {
                    P::POLICY.max_iterations
                }
            }
        };
        super::generic::phi::phi_internal::<Self, f64, P, false>(self, n, terms)
    }
}

impl<V: FloatVectorWithBits<Element = f64>> SpecializedRealSpecialMath<f64> for V
where
    V: TranscendentalMathWithPolicy<Element = f64>,
    V: SpecializedTranscendentalMath<f64>,
    // Pins the projection: a type parameter's `Primal` will not normalize through
    // the blanket impl on its own, and the table signatures need `Primal = Self`.
    V: thermite::math::PrimalProjection<Primal = V>,
{
    #[inline(always)]
    fn fresnel<P: Policy>(self) -> (Self, Self) {
        use crate::tables::fresnel as t;
        generic::fresnel::fresnel_with::<P, _, _, _, _, _, _>(
            self,
            t::X0_F64,
            t::MAP_F64,
            t::CUTOFF_F64,
            &t::CHEB_C_F64,
            &t::CHEB_S_F64,
            &t::AUX_P_F64,
            &t::AUX_Q_F64,
        )
    }

    #[inline(always)]
    fn sici<P: Policy>(self) -> (Self, Self) {
        use crate::tables::sici as t;
        generic::sici::sici_with::<P, _, _, _, _, _, _>(
            self,
            t::X0_F64,
            t::MAP_F64,
            t::CUTOFF_F64,
            &t::CHEB_SI_F64,
            &t::CHEB_CIN_F64,
            &t::AUX_P_F64,
            &t::AUX_Q_F64,
        )
    }

    // --- Spherical harmonics: the compile-time-table fast paths ---
    //
    // A concrete `f32`/`f64` element has a `ShConsts` table, which the generic
    // defaults cannot assume. Both overrides are guarded by `L <= MAX_SH_DEGREE`,
    // the extent of the stamped ladder, and fall back to the generic body above it.
    // A statically-false `if const` arm is dropped before monomorphization, so the
    // out-of-range table is never built.

    #[inline(always)]
    fn spherical_harmonics<P: Policy, const L: usize, const N: usize, const CS: bool>(
        x: Self,
        y: Self,
        z: Self,
        out: &mut [Self; N],
    ) {
        // Fully unrolled, constants folded into the instruction stream: no table is
        // materialized at all, so there is nothing to hoist out of a loop. Above
        // MAX_SH_DEGREE the kernel routes itself to the general path.
        sh_impl::<P, f64, Self, L, N, CS>(x, y, z, out);
    }

    #[inline(always)]
    fn spherical_harmonics_table<P: Policy, const L: usize, const N: usize, const CS: bool>(
        table: &mut ShTable<Self, N>,
    ) {
        if const { L <= MAX_SH_DEGREE } {
            // Every coefficient is already a compile-time constant of the right
            // phase, so building the runtime table is a splat per entry, with none of
            // the sqrt/divide work the generic default does.
            let src = &<f64 as ShConsts<L, N, CS>>::TABLE;

            let mut i = 0;
            while i < N {
                table.qmm[i] = Self::splat(src.qmm[i]);
                table.em[i] = Self::splat(src.em[i]);
                table.a[i] = Self::splat(src.a[i]);
                table.nb[i] = Self::splat(src.nb[i]);
                table.f[i] = Self::splat(src.f[i]);
                table.mf[i] = Self::splat(src.mf[i]);
                i += 1;
            }
        } else {
            sh_table_impl::<Self, L, N, CS>(table);
        }
    }

    #[inline(always)]
    fn bessel_i_ratio<P: Policy>(self, nu: Self) -> Self {
        generic::bessel::ratio::bessel_i_ratio_impl::<P, f64, Self>(self, nu)
    }

    #[inline(always)]
    fn inv_bessel_i_ratio<P: Policy>(self, nu: Self) -> Self {
        generic::bessel::ratio::inv_bessel_i_ratio_impl::<P, f64, Self>(self, nu)
    }

    #[inline(always)]
    fn bessel_i_ratio_1m<P: Policy>(self, nu: Self) -> Self {
        generic::bessel::ratio::bessel_i_ratio_1m_impl::<P, f64, Self>(self, nu)
    }

    #[inline(always)]
    fn inv_bessel_i_ratio_1m<P: Policy>(self, nu: Self) -> Self {
        generic::bessel::ratio::inv_bessel_i_ratio_1m_impl::<P, f64, Self>(self, nu)
    }

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
        if const { is_reference::<P>() } {
            // libm hands the sign back as an `i32`; this trait carries it as a float.
            return map1x2(self, |x| {
                let (v, s) = libm::lgamma_r(x);
                (v, s as f64)
            });
        }

        generic::gamma::lgamma_r_impl::<P, _, _, _>(self, &crate::tables::gamma::LANCZOS_F64)
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

    #[inline(always)]
    fn langevin<P: Policy>(self) -> Self {
        // The Worst/Medium tiers take the short table (see it for its error).
        if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
            generic::langevin::langevin_primal::<P, _, _, 11, false>(self, &LANGEVIN_SMALL_F64_LO).0
        } else {
            generic::langevin::langevin_primal::<P, _, _, 16, false>(self, &LANGEVIN_SMALL_F64).0
        }
    }

    #[inline(always)]
    fn langevin_1m<P: Policy>(self) -> Self {
        // The Worst/Medium tiers take the short table (see it for its error).
        if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
            generic::langevin::langevin_primal::<P, _, _, 11, true>(self, &LANGEVIN_SMALL_F64_LO).0
        } else {
            generic::langevin::langevin_primal::<P, _, _, 16, true>(self, &LANGEVIN_SMALL_F64).0
        }
    }

    // f64 refines with Halley (see the kernel docs).
    #[inline(always)]
    fn inv_langevin<P: Policy>(self) -> Self {
        generic::langevin::inv_langevin::<P, _, _, 16, 9, true, false>(self, &LANGEVIN_SMALL_F64, &LANGEVIN_SEED_F64)
    }

    #[inline(always)]
    fn inv_langevin_1m<P: Policy>(self) -> Self {
        generic::langevin::inv_langevin::<P, _, _, 16, 9, true, true>(self, &LANGEVIN_SMALL_F64, &LANGEVIN_SEED_F64)
    }

    // same form as f32
    #[inline(always)]
    fn gelu<P: Policy>(self, alpha: Self) -> Self {
        let x = self;

        let alpha_x = alpha * x;

        // GELU(x) = 0.5 * x * (1 + erf(ax / sqrt(2))) = 0.5 * x * erfc(-ax / sqrt(2))
        // O = false: skip the exp(-ax^2) byproduct that only the derivative needs.
        let mut unused = Self::EMPTY;
        let c = erf_d_internal::<V, P, true, false>(alpha_x * -Self::FRAC_1_SQRT_2, &mut unused);

        (x * Self::HALF) * c
    }
}

impl<V: FloatVectorWithBits<Element = f64>> SpecializedRealPrimalMath<f64> for V
where
    V: TranscendentalMathWithPolicy<Element = f64>,
    V: SpecializedTranscendentalMath<f64>,
    V: thermite::math::PrimalProjection<Primal = V>,
{
    #[inline(always)]
    fn langevin_d<P: Policy>(self) -> (Self, Self) {
        generic::langevin::langevin_primal::<P, _, _, 16, false>(self, &LANGEVIN_SMALL_F64)
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn spherical_harmonics_d<P: Policy, const L: usize, const N: usize, const CS: bool>(
        x: Self,
        y: Self,
        z: Self,
        out: &mut [Self; N],
        ddx: &mut [Self; N],
        ddy: &mut [Self; N],
        ddz: &mut [Self; N],
    ) {
        sh_d_impl::<P, f64, Self, L, N, CS>(x, y, z, out, ddx, ddy, ddz);
    }

    #[inline(always)]
    fn gelu_d<P: Policy>(self, alpha: Self) -> (Self, Self) {
        let x = self;

        let alpha_x = alpha * x;

        // 0.5 * x * erfc(-ax / sqrt(2))
        let mut exp_neg_ax2 = Self::EMPTY;
        let c = erf_d_internal::<V, P, true, true>(alpha_x * -Self::FRAC_1_SQRT_2, &mut exp_neg_ax2);

        let half_c = c * Self::HALF; // 0.5 * (1 + erf(ax/sqrt(2)))

        let y = x * half_c;
        let dy = if matches!(V::HAS_NATIVE_FMA, thermite::tribool::True) {
            (alpha_x * Self::FRAC_1_SQRT_TAU).mul_add(exp_neg_ax2, half_c)
        } else {
            half_c + alpha_x * Self::FRAC_1_SQRT_TAU * exp_neg_ax2
        };

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

    // Past |x| = 1.34e154 (and at infinity) x^2 overflows, the rational below is inf/inf
    // and the whole thing is NaN. erfc(27.3) has already underflowed and erf(6) is
    // exactly 1, so clamping at 32 changes no finite result. Compare-and-select rather
    // than `min` so a NaN input stays NaN on every backend.
    if const { P::POLICY.check_overflow } {
        let cap: V = thermite::const_splat!(f64: 32.0);
        x = x.cmp_gt(cap).select(cap, x);
    }

    // if ignoring denormals (and not clamping), just multiply x0 by itself to save like one
    // cycle, instead of waiting on abs(), otherwise use the denormal-flushed x value
    let x2 = if const { matches!(P::POLICY.denormal_behavior, DenormalBehavior::Ignore) && !P::POLICY.check_overflow } {
        x0 * x0
    } else {
        x * x
    };

    // LLVM will still start on exp and interleave it with the below operations.
    let e = (-x2).exp_p::<P>();

    // `x * x` rounds once, and the exp turns that relative `eps` into a relative `x^2 eps`
    // of the result: 47 ulp at x = 14, 237 at 24, the whole of erfc's tail error (measured
    // flat at 2.7 ulp over 0..27 once it is gone, `tests/erfc_tail.rs`).
    //
    // With a hardware FMA the residual `lo = x*x - x2` is exact and
    // `e^{-(x2 + lo)} = e^{-x2}(1 - lo)` to first order, `lo` being below `eps * x2 < 1e-13`
    // wherever erfc is representable. Two instructions, at every tier. Not for the emulated
    // FMA: the residual of an unfused product is meaningless, and the polyfill is not for
    // shipped kernels.
    //
    // Without one, `Best` takes fdlibm's split instead: `z` is `x` with its low 27 mantissa
    // bits cleared, so `z * z` is exact, and `e^{-x^2} = e^{-z^2} e^{-t}` with
    // `t = (x - z)(x + z)`, both factors exact by Sterbenz, `t < x^2 2^-25 < 2e-5` over the
    // whole range, so four Taylor terms of `e^{-t}` are 1e-20. No division and no FMA. It
    // replaced six independent divisions that measured no gain in any band.
    let e = if const { matches!(V::HAS_NATIVE_FMA, thermite::tribool::True) } {
        let lo = x.mul_sub(x, x2);
        lo.nmul_add(e, e)
    } else if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
        let z = V::from_bits(x.into_bits::<V::Bits>() & thermite::const_splat!(u64: 0xffff_ffff_f800_0000));
        let t = (x - z) * (x + z);
        let e = (-(z * z)).exp_p::<P>();
        e * t.poly_n_p::<P, _>(&[1.0, -1.0, 0.5, -1.0 / 6.0])
    } else {
        e
    };

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

    // One division at every tier. The six independent divisions `Best` used to take here
    // measured identical to this in every band on both lowerings (`tests/erfc_tail.rs`,
    // 2026-09-01): the product's rounding is not where erfc's error was.
    let m = {
        let n = (a0 * b0) * (c0 * d0) * (e0 * f0);
        let d = (a1 * b1) * (c1 * d1) * (e1 * f1);
        n / d
    };

    if O {
        // write this right before we use e normally, so LLVM can interleave exp with the above
        *out_exp_neg_x2 = e;
    }

    if !C {
        let y = e.nmul_adde(m, V::ONE) ^ sign;

        // `1 - m e^{-x^2}` carries a fixed absolute error of about an ulp of 1, so below
        // |x| ~ 1e-3 it has no relative accuracy left (erf(0) itself came out 2.2e-16). At
        // `Best` and above, small arguments take fdlibm's erf(x) = x + x R(x^2)/S(x^2) on
        // |x| < 0.84375 (libm s_erf.c coefficients, 2^-59 relative), which is exact at zero
        // and odd by construction. The f32 kernel has carried the same arm from `Average`.
        if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            let small = x.cmp_lt(thermite::const_splat!(f64: 0.84375));

            if const { P::POLICY.avoid_branching } || small.any() {
                let rs = x2.poly_rational_n_p::<P, _, _>(
                    &[
                        1.28379167095512558561e-01,
                        -3.25042107247001499370e-01,
                        -2.84817495755985104766e-02,
                        -5.77027029648944159157e-03,
                        -2.37630166566501626084e-05,
                    ],
                    &[
                        1.0,
                        3.97917223959155352819e-01,
                        6.50222499887672944485e-02,
                        5.08130628187576562776e-03,
                        1.32494738004321644526e-04,
                        -3.96022827877536812320e-06,
                    ],
                );

                return small.select(x0.mul_adde(rs, x0), y);
            }
        }

        y
    } else if const { matches!(V::HAS_NATIVE_FMA, thermite::tribool::True) } {
        // exploit instruction-level parallelism if FMA is available
        x0.select_negative(m.nmul_add(e, V::TWO), m * e)
    } else {
        let y = m * e;

        x0.select_negative(V::TWO - y, y)
    }
}

/// Every default applies: `expint` on the real line is what they were written for.
impl<V: FloatVectorWithBits<Element = f64>> super::ExpIntDetails<f64, V> for V {}

/// Minimax fit of `L(x)/x` as a polynomial in `x^2` on `[0, 2]`, relative error
/// `6.5e-17` after rounding (`crates/thermite-special/scripts/langevin_coeffs.py`).
const LANGEVIN_SMALL_F64: [f64; 16] = [
    0.3333333333333333,
    -0.022222222222221866,
    0.002116402116394456,
    -0.0002116402115749962,
    2.1377798863187195e-05,
    -2.1644034853512783e-06,
    2.1925805178692086e-07,
    -2.2212830510921946e-08,
    2.2491902490670497e-09,
    -2.2699964972216279e-10,
    2.258858881319397e-11,
    -2.149441406576282e-12,
    1.8363701272379474e-13,
    -1.27083611669092e-14,
    6.073730625469803e-16,
    -1.4527886936695518e-17,
];

/// Minimax fit of `L^-1(y) (1 - y^2) / y` as a polynomial in `y^2` on `[0, 0.85^2]`,
/// relative error `1.1e-6`. The inverse's Halley seed below the `1/(1-y)` tail. Deg 8
/// rather than f32's deg 4 so that one cubic step (constant < 0.07) lands under f64's u.
const LANGEVIN_SEED_F64: [f64; 9] = [
    3.0000033409892763,
    -1.200575454653041,
    -0.08655290656271598,
    -0.11229572780500324,
    0.9977556478992905,
    -2.159977066903404,
    2.891255158804264,
    -0.7894725857798888,
    -0.6113193395162252,
];

/// The `Worst`/`Medium` forward table: same fit as [`LANGEVIN_SMALL_F64`] at degree 10,
/// relative error `1.9e-12` (the Medium tier's tolerance is 1e4 eps), five FMAs cheaper.
/// The inverse keeps the full table at every tier, since its step is dominated by the
/// exp and the division and its Medium tier is documented as full precision.
const LANGEVIN_SMALL_F64_LO: [f64; 11] = [
    0.3333333333327056,
    -0.02222222218374748,
    0.002116401724776868,
    -0.00021163864787629512,
    2.1374572036431745e-05,
    -2.160476424670753e-06,
    2.162294603438737e-07,
    -2.0672900635769838e-08,
    1.7227111614831327e-09,
    -1.0503475016377067e-10,
    3.3022089667780975e-12,
];

/// Order dispatch for the modified Bessel entry points. `N` is a const parameter, so the
/// `if const` collapses to one arm and the unused table is never built.
///
/// Orders 0 and 1 are closed forms. Everything above seeds from the order-0 form and walks
/// the ratio recurrence down. All three arms are selected at compile time, so a call site
/// pays for exactly one.
#[inline(always)]
fn bessel_i_dispatch<P: Policy, V, const N: i32, const SCALED: bool>(x: V) -> V
where
    V: thermite::vector::FloatVector<Element = f64> + thermite::math::TranscendentalMathWithPolicy,
{
    if const { N == 0 } {
        generic::bessel::ik::bessel_i0_impl::<P, _, _, _, _, SCALED>(x, &crate::tables::bessel::BESSEL_I0_F64)
    } else if const { N.unsigned_abs() == 1 } {
        generic::bessel::ik::bessel_i1_impl::<P, _, _, _, _, SCALED>(x, &crate::tables::bessel::BESSEL_I1_F64)
    } else {
        // Orders past 1 seed from the order-0 closed form and walk the ratio recurrence down.
        generic::bessel::ik::bessel_in_impl::<P, f64, _, _, _, _, N, SCALED>(x, &crate::tables::bessel::BESSEL_I0_F64)
    }
}

/// Order dispatch for the modified Bessel functions of the second kind.
///
/// Orders 0 and 1 are closed forms. Above that the recurrence runs **upward**, which is the
/// opposite of the `I` family and is stable for exactly that reason: `K` is the dominant
/// solution. Both `K` kernels also need the `I` tables, because their small arms are
/// `P(x^2) - ln(x) I_0(x)` and `R(x^2) x + 1/x + ln(x) I_1(x)`.
#[inline(always)]
fn bessel_k_dispatch<P: Policy, V, const N: i32, const SCALED: bool>(x: V) -> V
where
    V: thermite::vector::FloatVector<Element = f64> + thermite::math::TranscendentalMathWithPolicy,
{
    use crate::tables::bessel::{BESSEL_I0_F64, BESSEL_I1_F64, BESSEL_K0_F64, BESSEL_K1_F64};
    if const { N == 0 } {
        generic::bessel::ik::bessel_k0_impl::<P, f64, _, _, _, _, _, _, _, _, SCALED>(x, &BESSEL_K0_F64, &BESSEL_I0_F64)
    } else if const { N.unsigned_abs() == 1 } {
        generic::bessel::ik::bessel_k1_impl::<P, f64, _, _, _, _, _, _, _, _, SCALED>(x, &BESSEL_K1_F64, &BESSEL_I1_F64)
    } else {
        // The recurrence takes the two seeds, not the tables. Each closed form infers its
        // own array lengths here, at the one place that already names them concretely.
        let k0 = generic::bessel::ik::bessel_k0_impl::<P, f64, _, _, _, _, _, _, _, _, SCALED>(
            x,
            &BESSEL_K0_F64,
            &BESSEL_I0_F64,
        );
        let k1 = generic::bessel::ik::bessel_k1_impl::<P, f64, _, _, _, _, _, _, _, _, SCALED>(
            x,
            &BESSEL_K1_F64,
            &BESSEL_I1_F64,
        );
        generic::bessel::ik::bessel_kn_recur::<f64, _, N>(x, k0, k1).1
    }
}

/// `(I_N, I_N prime)`, sharing the order-`N-1` value the recurrence already produces.
///
/// `I_N' = I_{N-1} - (N/x) I_N`, and at `N = 0` the second term vanishes because
/// `I_{-1} = I_1`, so one formula covers every order, with the `N = 0` case written out to
/// keep `0/x` from becoming `0/0` at the origin.
///
/// Scaled adds one term: `d/dx e^{-|x|}f = e^{-|x|}(f' - sgn(x) f)`.
#[inline(always)]
fn bessel_i_deriv_dispatch<P: Policy, V, const N: i32, const SCALED: bool>(x: V) -> (V, V)
where
    V: thermite::vector::FloatVector<Element = f64> + thermite::math::TranscendentalMathWithPolicy,
{
    let (prev, v) = if const { N == 0 } {
        (
            bessel_i_dispatch::<P, V, 1, SCALED>(x),
            bessel_i_dispatch::<P, V, 0, SCALED>(x),
        )
    } else if const { N.unsigned_abs() == 1 } {
        (
            bessel_i_dispatch::<P, V, 0, SCALED>(x),
            bessel_i_dispatch::<P, V, 1, SCALED>(x),
        )
    } else {
        generic::bessel::ik::bessel_in_pair_impl::<P, f64, _, _, _, _, N, SCALED>(
            x,
            &crate::tables::bessel::BESSEL_I0_F64,
        )
    };

    let mut d = if const { N == 0 } {
        prev
    } else {
        prev - v * (V::splat(N.unsigned_abs() as f64) / x)
    };
    if const { SCALED } {
        d -= v.copysign(x);
    }
    (v, d)
}

/// `(K_N, K_N prime)`. `K_N' = -K_{N-1} - (N/x) K_N`, and `K_{-1} = K_1`.
///
/// Scaled subtracts rather than adds, since the scaling runs the other way:
/// `d/dx e^{x}f = e^{x}(f' + f)`.
#[inline(always)]
fn bessel_k_deriv_dispatch<P: Policy, V, const N: i32, const SCALED: bool>(x: V) -> (V, V)
where
    V: thermite::vector::FloatVector<Element = f64> + thermite::math::TranscendentalMathWithPolicy,
{
    let (prev, v) = if const { N == 0 } {
        (
            bessel_k_dispatch::<P, V, 1, SCALED>(x),
            bessel_k_dispatch::<P, V, 0, SCALED>(x),
        )
    } else if const { N.unsigned_abs() == 1 } {
        (
            bessel_k_dispatch::<P, V, 0, SCALED>(x),
            bessel_k_dispatch::<P, V, 1, SCALED>(x),
        )
    } else {
        // The upward recurrence walks THROUGH order N-1 on its way to N, so the pair costs
        // nothing beyond returning it.
        let k0 = generic::bessel::ik::bessel_k0_impl::<P, f64, _, _, _, _, _, _, _, _, SCALED>(
            x,
            &crate::tables::bessel::BESSEL_K0_F64,
            &crate::tables::bessel::BESSEL_I0_F64,
        );
        let k1 = generic::bessel::ik::bessel_k1_impl::<P, f64, _, _, _, _, _, _, _, _, SCALED>(
            x,
            &crate::tables::bessel::BESSEL_K1_F64,
            &crate::tables::bessel::BESSEL_I1_F64,
        );
        generic::bessel::ik::bessel_kn_recur::<f64, _, N>(x, k0, k1)
    };

    let mut d = if const { N == 0 } {
        -prev
    } else {
        -prev - v * (V::splat(N.unsigned_abs() as f64) / x)
    };
    if const { SCALED } {
        d += v;
    }
    (v, d)
}
