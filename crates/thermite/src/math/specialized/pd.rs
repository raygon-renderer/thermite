use crate::divider::Divider;
use crate::vector::ops::{AddMasked as _, MulMasked as _, SubMasked as _};
use core::f64::consts::{LN_10, LOG2_E, SQRT_2};

use super::*;

// The `PrimalProjection<Primal = V>` pin: the rigid `PrimalProjection` supertrait
// of `SpecializedCoreMath` shadows the fixpoint blanket impl on a generic `V`, so
// without it `V::Primal` would not normalize to `V` in the `poly_primal` body.
impl<V: FloatVectorWithBits<Element = f64> + PrimalProjection<Primal = V>> SpecializedCoreMath<f64> for V {
    /// `Primal = Self`, so the coefficients are already this vector type, which means
    /// the ILP lowering [`poly`](SpecializedCoreMath::poly) uses applies unchanged, and
    /// the primal-Horner default would be a straight downgrade for real vectors.
    ///
    /// The one difference from `poly` is that these coefficients arrive pre-splatted, so
    /// the per-term `splat` disappears too.
    #[inline(always)]
    fn poly_primal<P: Policy, N: ArrayLength>(self, coeffs: &GenericArray<Self::Primal, N>) -> Self {
        super::generic::poly_primal_internal::<V, P, N>(self, coeffs)
    }

    /// The reverse-order twin of [`poly_primal`](SpecializedCoreMath::poly_primal), with
    /// the same reasoning: pre-splatted coefficients over the ILP lowering of
    /// [`poly_rev`](SpecializedCoreMath::poly_rev).
    #[inline(always)]
    fn poly_rev_primal<P: Policy, N: ArrayLength>(self, coeffs: &GenericArray<Self::Primal, N>) -> Self {
        super::generic::poly_rev_primal_internal::<V, P, N>(self, coeffs)
    }

    #[inline(always)]
    fn inverse_sqrt<P: Policy>(self) -> Self {
        super::generic::inverse_sqrt_internal::<V, f64, P>(self)
    }
}

impl<V: FloatVectorWithBits<Element = f64>> SpecializedPrimalMath<f64> for V {}

impl<V: FloatVectorWithBits<Element = f64>> SpecializedRealMath<f64> for V {
    #[inline(always)]
    fn atan2<P: Policy>(self, x: Self) -> Self {
        atan_internal::<Self, P, true>(self, x)
    }

    #[inline(always)]
    fn wrap_angle<P: Policy>(self) -> Self {
        let x = self;
        let n = ((x + Self::PI) * (Self::FRAC_1_PI * Self::HALF)).floor();

        if const { P::POLICY.precision.le(PrecisionPolicy::Average) } {
            return n.nmul_adde(Self::TAU, x);
        }

        // Cody-Waite against the TRUE 2 pi, not against fl(2 pi). The dominant error at
        // large |x| is not the rounding of `n * TAU` (a fused step makes that
        // single-rounded for free), it is that TAU is only 2 pi to half an ulp, so even a
        // perfectly fused `x - n * TAU` drifts by `n * 2.45e-16`. That is 0.04 rad by
        // x = 1e15, a WRONG angle once it crosses the +-pi seam, which is exactly what
        // the old FMA branch did.
        let mut r = if const { Self::HAS_TRUE_FMA } {
            // Two fused steps: tau_hi is fl(2 pi) == TAU (full mantissa, so the FMA
            // multiplies it exactly), tau_lo the next 53 bits, together 2 pi to ~1e-32
            // relative. The estimating forms ARE single fused instructions on this
            // hardware, so emulated FMA is never used.
            let tau_hi: V = crate::const_splat!(f64: hexf::hexf64!("0x1.921fb54442d18p+2"));
            let tau_lo: V = crate::const_splat!(f64: hexf::hexf64!("0x1.1a62633145c07p-52"));

            let r = n.nmul_adde(tau_hi, x);
            n.nmul_adde(tau_lo, r)
        } else {
            // No FMA: exact-by-construction split, plain mul/sub only. The 24-bit parts
            // make `n * tau_a` and `n * tau_b` exact products while bits(n) <= 29
            // (24 + 29 = 53), the first subtraction is exact by Sterbenz (x and
            // n * tau_a agree to within a factor of two), and the remaining subtractions
            // are correctly rounded at the RESULT's magnitude, so the whole chain lands
            // within ~1.5 ulp of pi. The full-mantissa tail takes the constant to
            // 2 pi * 2^-101.
            //
            // Valid to |x| <~ 2^29 * 2 pi ~ 3.4e9. Past that the products start rounding
            // and accuracy decays gradually toward the naive form's (3.3e-6 rad at
            // 1e11), still confined to [-pi, pi) by the repair below. The FMA path above
            // reaches ~2^53 * pi instead.
            let tau_a: V = crate::const_splat!(f64: hexf::hexf64!("0x1.921fb60000000p+2"));
            let tau_b: V = crate::const_splat!(f64: hexf::hexf64!("-0x1.777a5c0000000p-23"));
            let tau_c: V = crate::const_splat!(f64: hexf::hexf64!("-0x1.ee59d9cceba40p-48"));

            ((x - n * tau_a) - n * tau_b) - n * tau_c
        };

        // The floor can land one period off: its quotient rounds (`x + pi` alone costs up
        // to half an ulp of x), so near a seam `n` is off by one and `r` by 2 pi. One
        // masked add/sub each way repairs it. Beyond |x| ~ 2^53 * pi the quotient's own
        // ulp exceeds one period and the result degrades to "some representative of an
        // angle". At that magnitude ulp(x) > 2 pi, so the input no longer determines an
        // angle anyway.
        r = r.add_c(r.cmp_lt(-Self::PI), Self::TAU);
        r = r.sub_c(r.cmp_ge(Self::PI), Self::TAU);

        r
    }
}

#[rustfmt::skip]
impl<V: FloatVectorWithBits<Element = f64>> SpecializedSpatialMath<f64> for V {
    #[inline(always)] fn l2_norm_squared<P: Policy>(self) -> Self { self * self }
    #[inline(always)] fn l2_norm<P: Policy>(self) -> Self { self.abs() }
    #[inline(always)] fn l1_norm<P: Policy>(self) -> Self { self.abs() }
}

impl<V: FloatVectorWithBits<Element = f64>> SpecializedTranscendentalMath<f64> for V {
    #[inline(always)]
    fn sinc<P: Policy>(self) -> Self {
        super::generic::sinc_internal::<V, f64, P>(self)
    }

    #[inline(always)]
    fn sinc_pi<P: Policy>(self) -> Self {
        super::generic::sinc_pi_internal::<V, f64, P>(self)
    }

    #[inline(always)]
    fn log_n<P: Policy, const N: usize>(self) -> Self {
        super::generic::log_n_internal::<V, f64, P, N>(self)
    }

    #[inline(always)]
    fn sin_cos<P: Policy>(self) -> (Self, Self) {
        sincos_d_internal::<P, V, false>(self)
    }

    #[inline(always)]
    fn sincos_pi<P: Policy>(self) -> (Self, Self) {
        sincos_d_internal::<P, V, true>(self)
    }

    #[inline(always)]
    fn sinh_cosh<P: Policy>(self) -> (Self, Self) {
        let x0 = self;
        let x = x0.abs().flush_denormals::<P>();
        let y = x.exph_p::<P>();
        let qy = V::FRAC_1_4 / y;

        let mut sinh = y - qy;
        let cosh = y + qy;

        let x_small = x.cmp_le(V::ONE);

        // if any are small, use a polynomial approximation
        if const { P::POLICY.avoid_branching } || x_small.any() {
            let x2 = x * x;

            #[rustfmt::skip]
            let y1 = x2.poly_rational_p::<P, _, _>(
                &[
                    -3.51754964808151394800E5,
                    -1.15614435765005216044E4,
                    -1.63725857525983828727E2,
                    -7.89474443963537015605E-1,
                ],
                &[
                    -2.11052978884890840399E6,
                    3.61578279834431989373E4,
                    -2.77711081420602794433E2,
                    1.0,
                ],
            ).mul_adde(x * x2, x);

            sinh = x_small.select(y1, sinh);
        }

        (sinh.mul_sign(x0), cosh)
    }

    #[inline(always)]
    fn sinh<P: Policy>(self) -> Self {
        let x0 = self;
        let x = x0.abs().flush_denormals::<P>();

        let x_small = x.cmp_le(V::ONE);

        let mut y2 = V::EMPTY;

        if const { P::POLICY.avoid_branching } || !x_small.all() {
            y2 = x.exph_p::<P>();
            y2 -= V::FRAC_1_4 / y2;

            // if we don't care about small x, we can skip the next branch
            if const { P::POLICY.avoid_precision_branches() } {
                return y2.mul_sign(x0);
            }
        }

        if const { P::POLICY.avoid_branching } || x_small.any() {
            let x2 = x * x;

            #[rustfmt::skip]
            let y1 = x2.poly_rational_p::<P, _, _>(
                &[
                    -3.51754964808151394800E5,
                    -1.15614435765005216044E4,
                    -1.63725857525983828727E2,
                    -7.89474443963537015605E-1,
                ],
                &[
                    -2.11052978884890840399E6,
                    3.61578279834431989373E4,
                    -2.77711081420602794433E2,
                    1.0,
                ],
            ).mul_adde(x * x2, x);

            y2 = x_small.select(y1, y2);
        }

        y2.mul_sign(x0)
    }

    #[inline(always)]
    fn cosh<P: Policy>(self) -> Self {
        let y = self.abs().exph_p::<P>();
        y + V::FRAC_1_4 / y
    }

    #[inline(always)]
    fn tanh<P: Policy>(self) -> Self {
        let x0 = self;
        let x = x0.abs().flush_denormals::<P>();

        let x_small = x.cmp_le(crate::const_splat!(f64: 0.625));

        let mut y2 = V::EMPTY;

        if const { P::POLICY.avoid_branching } || !x_small.all() {
            let h = (x + x).exph_p::<P>();
            y2 = (h - V::HALF) / (h + V::HALF);

            if const { P::POLICY.check_overflow } {
                y2 = x.cmp_gt(crate::const_splat!(f64: 350.0)).select(V::ONE, y2);
            }

            if const { P::POLICY.avoid_precision_branches() } {
                return y2.mul_sign(x0);
            }
        }

        if const { P::POLICY.avoid_branching } || x_small.any() {
            let x2 = x * x;

            #[rustfmt::skip]
            let y1 = x2.poly_rational_p::<P, _, _>(
                &[
                    -1.61468768441708447952E3,
                    -9.92877231001918586564E1,
                    -9.64399179425052238628E-1,
                ],
                &[
                     4.84406305325125486048E3,
                    2.23548839060100448583E3,
                    1.12811678491632931402E2,
                    1.0,
                ],
            ).mul_adde(x * x2, x);

            y2 = x_small.select(y1, y2);
        }

        y2.mul_sign(x0)
    }

    #[inline(always)]
    fn asin<P: Policy>(self) -> Self {
        asin_internal::<Self, P, false>(self)
    }

    #[inline(always)]
    fn acos<P: Policy>(self) -> Self {
        asin_internal::<Self, P, true>(self)
    }

    #[inline(always)]
    fn atan<P: Policy>(self) -> Self {
        atan_internal::<Self, P, false>(self, V::ZERO)
    }

    #[inline(always)]
    fn asinh<P: Policy>(self) -> Self {
        let x0 = self;
        let x = x0.abs().flush_denormals::<P>();
        let x2 = x * x;

        let x_small = x.cmp_le(crate::const_splat!(f64: 0.533));

        let mut y2 = V::EMPTY;

        if const { P::POLICY.avoid_branching } || !x_small.all() {
            y2 = ((x2 + V::ONE).sqrt() + x).ln_p::<P>();

            if const { P::POLICY.check_overflow || !P::POLICY.avoid_precision_branches() } {
                let x_huge = x.cmp_gt(crate::const_splat!(f64: 1e20));

                if crate::unlikely(x_huge.any()) {
                    y2 = x_huge.select(x.ln_p::<P>() + V::LN_2, y2);
                }
            }
        }

        if const { P::POLICY.avoid_branching } || x_small.any() {
            let y1 = x2
                .poly_rational_p::<P, _, _>(
                    &[
                        -5.56682227230859640450E0,
                        -9.09030533308377316566E0,
                        -4.37390226194356683570E0,
                        -5.91750212056387121207E-1,
                        -4.33231683752342103572E-3,
                    ],
                    &[
                        3.34009336338516356383E1,
                        6.95722521337257608734E1,
                        4.86042483805291788324E1,
                        1.28757002067426453537E1,
                        1.0,
                    ],
                )
                .mul_adde(x * x2, x);

            y2 = x_small.select(y1, y2);
        }

        y2.mul_sign(x0)
    }

    #[inline(always)]
    fn acosh<P: Policy>(self) -> Self {
        let x0 = self.flush_denormals::<P>();
        let x1 = x0 - V::ONE;

        let x_small = x1.cmp_le(crate::const_splat!(f64: 0.49));

        let mut y2 = V::EMPTY;

        if const { P::POLICY.avoid_branching } || !x_small.all() {
            y2 = (x0.mul_sube(x0, V::ONE).sqrt() + x0).ln_p::<P>();

            if const { P::POLICY.check_overflow && !P::POLICY.avoid_precision_branches() } {
                let x_huge = x1.cmp_gt(crate::const_splat!(f64: 1e20));

                if crate::unlikely(x_huge.any()) {
                    y2 = x_huge.select(x0.ln_p::<P>() + V::LN_2, y2);
                }
            }

            if const { P::POLICY.avoid_precision_branches() } {
                // certain overflow checks can still be important even if precision is not
                if const { P::POLICY.check_overflow } {
                    y2 = x0.cmp_lt(V::ONE).select(V::NAN, y2);
                }

                return y2;
            }
        }

        if const { P::POLICY.avoid_branching } || x_small.any() {
            let mut y1 = x1.sqrt()
                * x1.poly_rational_p::<P, _, _>(
                    &[
                        1.10855947270161294369E5,
                        1.08102874834699867335E5,
                        3.43989375926195455866E4,
                        3.94726656571334401102E3,
                        1.18801130533544501356E2,
                    ],
                    &[
                        7.83869920495893927727E4,
                        8.29725251988426222434E4,
                        2.97683430363289370382E4,
                        4.15352677227719831579E3,
                        1.86145380837903397292E2,
                        1.0,
                    ],
                );

            if const { P::POLICY.check_overflow } {
                y1 = x0.cmp_lt(V::ONE).select(V::NAN, y1);
            }

            y2 = x_small.select(y1, y2);
        }

        y2
    }

    #[inline(always)]
    fn atanh<P: Policy>(self) -> Self {
        let x0 = self;
        let x = x0.abs().flush_denormals::<P>();

        let x_small = x.cmp_le(V::HALF);

        let mut y2 = V::EMPTY;

        if const { P::POLICY.avoid_branching } || !x_small.all() {
            y2 = ((V::ONE + x) / (V::ONE - x)).ln_p::<P>().scale(0.5);

            if const { P::POLICY.check_overflow } {
                let y3 = x.cmp_eq(V::ONE).select(V::INFINITY, V::NAN);
                y2 = x.cmp_ge(V::ONE).select(y3, y2);
            }

            if const { P::POLICY.avoid_precision_branches() } {
                return y2.mul_sign(x0);
            }
        }

        if const { P::POLICY.avoid_branching } || x_small.any() {
            let x2 = x * x;

            let y1 = x2
                .poly_rational_p::<P, _, _>(
                    &[
                        -3.09092539379866942570E1,
                        6.54566728676544377376E1,
                        -4.61252884198732692637E1,
                        1.20426861384072379242E1,
                        -8.54074331929669305196E-1,
                    ],
                    &[
                        -9.27277618139601130017E1,
                        2.52006675691344555838E2,
                        -2.49839401325893582852E2,
                        1.08938092147140262656E2,
                        -1.95638849376911654834E1,
                        1.0,
                    ],
                )
                .mul_adde(x * x2, x);

            y2 = x_small.select(y1, y2);
        }

        y2.mul_sign(x0)
    }

    #[inline(always)]
    fn exp<P: Policy>(self) -> Self {
        exp_d_internal::<Self, P, EXP_MODE_EXP>(self)
    }

    #[inline(always)]
    fn exph<P: Policy>(self) -> Self {
        exp_d_internal::<Self, P, EXP_MODE_EXPH>(self)
    }

    #[inline(always)]
    fn exp2<P: Policy>(self) -> Self {
        exp_d_internal::<Self, P, EXP_MODE_POW2>(self)
    }

    #[inline(always)]
    fn exp10<P: Policy>(self) -> Self {
        exp_d_internal::<Self, P, EXP_MODE_POW10>(self)
    }

    #[inline(always)]
    fn exp_m1<P: Policy>(self) -> Self {
        exp_d_internal::<Self, P, EXP_MODE_EXPM1>(self)
    }

    #[inline(always)]
    fn exp2_m1<P: Policy>(self) -> Self {
        exp_d_internal::<Self, P, EXP_MODE_POW2M1>(self)
    }

    #[inline(always)]
    fn exp10_m1<P: Policy>(self) -> Self {
        exp_d_internal::<Self, P, EXP_MODE_POW10M1>(self)
    }

    #[inline(always)]
    fn powf<P: Policy>(self, y: Self) -> Self {
        let x0 = self;

        // define constants
        let ln2d_hi = crate::const_splat!(f64: 0.693145751953125); // log(2) in extra precision, high bits
        let ln2d_lo = crate::const_splat!(f64: 1.42860682030941723212E-6); // low bits of log(2)

        let x1 = x0.abs().flush_denormals::<P>();

        let mut x = fraction2(x1);

        let blend = x.cmp_gt(crate::const_splat!(f64: SQRT_2 / 2.0));

        x.add_assign_c(!blend, x); // conditional assign, only if blend is false
        x -= V::ONE;

        let x2 = x * x;

        #[rustfmt::skip]
        let lg1 = (x2 * x) * x.poly_rational_p::<P, _, _>(
            &[
                2.0039553499201281259648E1,
                5.7112963590585538103336E1,
                6.0949667980987787057556E1,
                2.9911919328553073277375E1,
                6.5787325942061044846969E0,
                4.9854102823193375972212E-1,
                4.5270000862445199635215E-5,
            ],
            &[
                6.0118660497603843919306E1,
                2.1642788614495947685003E2,
                3.0909872225312059774938E2,
                2.2176239823732856465394E2,
                8.3047565967967209469434E1,
                1.5062909083469192043167E1,
                1.0,
            ],
        );

        let ef = exponent_f(x1).add_c(blend, V::ONE); // conditional add, only if blend is true

        // multiply exponent by y, nearest integer e1 goes into exponent of result, remainder yr is added to log
        let e1 = (ef * y).round();
        let yr = ef.mul_sube(y, e1); // calculate remainder yr. precision very important here

        // add initial terms to expansion
        let lg = V::HALF.nmul_adde(x2, x) + lg1; // lg = (x - 0.5f * x2) + lg1;

        // calculate rounding errors in lg
        // rounding error in multiplication 0.5*x*x
        let x2err = (V::HALF * x).mul_sube(x, V::HALF * x2);

        // rounding error in additions and subtractions
        let lgerr = V::HALF.mul_adde(x2, lg - x) - lg1; // lgerr = ((lg - x) + 0.5f * x2) - lg1;

        // extract something for the exponent
        let e2 = (lg * y).scale(FloatConsts::LOG2_E).round();

        // subtract this from lg, with extra precision
        let mut v = e2.nmul_adde(ln2d_lo, lg.mul_sube(y, e2 * ln2d_hi));

        // add remainder from ef * y
        v = yr.mul_adde(V::LN_2, v); // v += yr * VM_LN2;

        // correct for previous rounding errors
        v = (lgerr + x2err).nmul_adde(y, v); // v -= (lgerr + x2err) * y;

        // extract something for the exponent if possible
        let mut x = v;
        let e3 = x.scale(FloatConsts::LOG2_E).round();

        // high precision multiplication not needed here because abs(e3) <= 1
        x = e3.nmul_adde(V::LN_2, x); // x -= e3 * VM_LN2;

        // Taylor coefficients for exp function, 1/n!
        let z = x.poly_rev_p::<P, _>(&[
            1.0 / 6227020800.0,
            1.0 / 479001600.0,
            1.0 / 39916800.0,
            1.0 / 3628800.0,
            1.0 / 362880.0,
            1.0 / 40320.0,
            1.0 / 5040.0,
            1.0 / 720.0,
            1.0 / 120.0,
            1.0 / 24.0,
            1.0 / 6.0,
            1.0 / 2.0,
            1.0, // 1x
            1.0, // + 1
        ]);

        // contributions to exponent
        let ee = e1 + e2 + e3;
        let ei: V::SignedBits = ee.fast_cast();

        // biased exponent of result:
        let ej = ei + (V::SignedBits::from_bits(z.abs()) >> 52);

        // add exponent by signed integer addition
        let mut z = V::from_bits(V::SignedBits::from_bits(z) + (ei << 52));

        if const { !P::POLICY.check_overflow } {
            // x^0 == 1, kept even on the fast path (exponent-split otherwise
            // leaves x's exponent in for y == 0).
            return y.cmp_eq(V::ZERO).select(V::ONE, z);
        }

        // check exponent for overflow and underflow
        let overflow =
            ej.cmp_ge(V::SignedBits::splat(0x07FF)).cast::<V::Mask>() | ee.cmp_gt(crate::const_splat!(f64: 3000.0));
        let underflow =
            ej.cmp_le(V::SignedBits::splat(0x0000)).cast::<V::Mask>() | ee.cmp_lt(crate::const_splat!(f64: -3000.0));

        // check for special cases
        let xfinite = x0.is_finite();
        let yfinite = y.is_finite();
        let efinite = ee.is_finite();

        let xzero = x0.is_zero_or_subnormal();
        let xsign = x0.is_negative();

        if crate::unlikely((overflow | underflow).any()) {
            z = underflow.select(V::ZERO, z);
            z = overflow.select(V::INFINITY, z);
        }

        let yzero = y.cmp_eq(V::ZERO);
        let yneg = y.cmp_lt(V::ZERO);

        // pow_case_x0
        z = xzero.select(yneg.select(V::INFINITY, yzero.select(V::ONE, V::ZERO)), z);

        let mut yodd = V::ZERO;

        if xsign.any() {
            let yint = y.cmp_eq(y.round());
            yodd = V::from_bits(y.into_bits::<V::Bits>() << 63);

            let z1 = yint.select(z | yodd, x0.cmp_eq(V::ZERO).select(z, V::NAN));

            yodd = yint.select(yodd, V::ZERO);

            z = xsign.select(z1, z);
        }

        // x^0 == 1 for every (finite) x; line 534 only covered x == 0, so without
        // this the `not_special` fast return below leaks x's exponent for y == 0.
        z = yzero.select(V::ONE, z);

        let not_special = xfinite & yfinite & (efinite | xzero);

        if crate::likely(not_special.all()) {
            return z; // fast return
        }

        // handle special error cases: y infinite
        let z1 = (yfinite & efinite).select(
            z,
            x1.cmp_eq(V::ONE).select(
                V::ONE,
                (x1.cmp_gt(V::ONE) ^ y.is_negative()).select(V::INFINITY, V::ZERO),
            ),
        );

        // handle x infinite
        let z1 = xfinite.select(
            z1,
            yzero.select(
                V::ONE,
                yneg.select(
                    yodd & z,         // 0.0 with the sign of z from above
                    x1 | (x0 & yodd), // get sign of x0 only if y is odd integer
                ),
            ),
        );

        // Always propagate nan:
        // Deliberately differing from the IEEE-754 standard which has pow(0,nan)=1, and pow(1,nan)=1
        (x0.is_nan() | y.is_nan()).select(x0 + y, z1)
    }

    #[inline(always)]
    fn cbrt<P: Policy>(self) -> Self {
        let x = self.flush_denormals::<P>();

        let b1 = crate::const_splat!(u64: 715094163); // B1 = (1023-1023/3-0.03306235651)*2**20
        let b2 = crate::const_splat!(u64: 696219795); // B2 = (1023-1023/3-54/3-0.03306235651)*2**20
        let m = crate::const_splat!(u64: 0x7fffffff); // u32::MAX >> 1

        let x1p54 = x * Self::splat(f64::from_bits(0x4350000000000000)); // 0x1p54 === 2 ^ 54

        let hx0 = (x.into_bits::<V::Bits>() >> 32) & m;

        let x_small = hx0.cmp_lt(V::Bits::splat(0x00100000));

        let xs = x_small.select(x1p54, x); // note that this upcasts
        let b = x_small.select(b2, b1);

        let mut ui: V::Bits = xs.into_bits();
        let mut hx: V::Bits = (ui >> 32) & m;

        // NOTE: Using the branched divider with a constant
        // leads to better codegen when the branch is inlined.
        hx = hx / Divider::u64(3) + b;

        ui &= V::Bits::splat(1 << 63);
        ui |= hx << 32;

        let mut t = Self::from_bits(ui);

        let r = (t * t) * (t / x); // encourage ILP

        t *= r.poly_p::<P, _>(&[
            1.87595182427177009643,   /* 0x3ffe03e6, 0x0f61e692 */
            -1.88497979543377169875,  /* 0xbffe28e0, 0x92f02420 */
            1.621429720105354466140,  /* 0x3ff9f160, 0x4a49d6c2 */
            -0.758397934778766047437, /* 0xbfe844cb, 0xbee751d9 */
            0.145996192886612446982,  /* 0x3fc2b000, 0xd4e4edd7 */
        ]);

        ui = t.into_bits();
        ui = (ui + V::Bits::splat(0x80000000)) & V::Bits::splat(0xffffffffc0000000);
        t = Self::from_bits(ui);

        // Preserving denormals also forces the exact form: the fast one cubes
        // the root, and for a denormal `x` that lands back in the denormal
        // range with almost no precision left. `x / (t * t)` never does - see
        // the matching note in `ps.rs`.
        let r = if const {
            P::POLICY.precision.ge(PrecisionPolicy::Best)
                || !Self::HAS_TRUE_FMA
                || matches!(P::POLICY.denormal_behavior, DenormalBehavior::Preserve)
        } {
            // original form, 5 simple ops, 2 divisions. Every intermediate is
            // provably well-behaved (`t*t` exact, `t+t` exact, `r-t` exact,
            // |r| < |t|), which is what buys the <0.667 ulp bound - fdlibm,
            // musl and Rust's own libm all use exactly this and never form t^3.
            let xtt = x / (t * t);
            (xtt - t) / ((t + t) + xtt)
        } else if const { P::POLICY.precision.ge(PrecisionPolicy::Average) } {
            // fast form with the ratio scaled by 1/4: `2t^3 + x` is ~3x and
            // overflows to NaN for |x| > ~MAX/3, but both scale factors are
            // exact powers of two, so this is the same quotient with the
            // intermediates capped near 0.75x. Keeps the single division.
            //
            // The 1/4 is folded INTO the cube rather than applied after: the
            // guess is only ~5 bits, so an overshoot near MAX makes a plain
            // `t*t*t` overflow before the ratio is ever formed.
            let t3q = (t * t) * (t * Self::FRAC_1_4); // t^3/4
            let xq = x * Self::FRAC_1_4;

            (xq - t3q) / t3q.mul_add(Self::TWO, xq)
        } else {
            // fast form, 3 simple ops, 1 division, 1 fma. Overflows for the top
            // binade - accepted at Medium and below.
            let t3 = t * t * t;
            (x - t3) / t3.mul_add(Self::TWO, x)
        };

        t = r.mul_adde(t, t);

        if const { !P::POLICY.check_overflow } {
            return x.cmp_eq(Self::ZERO).select(x, t);
        }

        // cbrt(NaN, INF, +-0) is itself.
        //
        // `hx0` is the HIGH word of the f64, so the non-finite threshold is
        // 0x7FF00000 (f64's infinity there), NOT 0x7F800000 - that is f32's
        // pattern, and using it misclassified every finite value with exponent
        // >= 1017 (|x| > ~1.4e306, `f64::MAX` included) as non-finite and
        // handed it straight back. `>=` rather than `>` so infinity itself is
        // caught, matching musl.
        //
        // The zero test is on the float for the same reason: `hx0 == 0` is also
        // true for every subnormal below 2^-1043, which would return those
        // unchanged (`cbrt(5e-324)` giving `5e-324`). musl tests it after its
        // 2^54 rescale to avoid exactly this.
        let non_finite = hx0.cmp_ge(V::Bits::splat(0x7ff00000)).cast::<Self::Mask>();

        (non_finite | x.cmp_eq(Self::ZERO)).select(x, t)
    }

    #[inline(always)]
    fn ln<P: Policy>(self) -> Self {
        ln_d_internal::<Self, P, false>(self)
    }

    #[inline(always)]
    fn ln_1p<P: Policy>(self) -> Self {
        ln_d_internal::<Self, P, true>(self)
    }

    #[inline(always)]
    fn log2<P: Policy>(self) -> Self {
        ln_d_internal::<Self, P, false>(self).scale(FloatConsts::LOG2_E)
    }

    #[inline(always)]
    fn log10<P: Policy>(self) -> Self {
        ln_d_internal::<Self, P, false>(self).scale(FloatConsts::LOG10_E)
    }

    #[inline(always)]
    fn ln1m_expnx<P: Policy>(self) -> Self {
        // f64 has no fast rational approximation (unlike f32's Medium path), so below
        // Average this keeps the cheap naive form and its documented failure modes at the
        // extremes. Average and up get the accurate two-branch kernel.
        if const { P::POLICY.precision.ge(PrecisionPolicy::Average) } {
            return super::generic::ln1m_expnx_internal::<V, f64, P>(self);
        }

        (V::ONE - (-self).exp_p::<P>()).ln_p::<P>()
    }

    #[inline(always)]
    fn ln1m_expnx_ext<P: Policy>(self, _lnx: Self) -> Self {
        self.ln1m_expnx_p::<P>()
    }
}

#[inline(always)]
fn fraction2<V: FloatVectorWithBits<Element = f64>>(x: V) -> V {
    // set exponent to 0 + bias
    (x & crate::const_splat!(f64: f64::from_bits(0x000FFFFFFFFFFFFF)))
        | crate::const_splat!(f64: f64::from_bits(0x3FE0000000000000))
}

#[inline(always)]
fn exponent<V: FloatVectorWithBits<Element = f64>>(x: V) -> V::SignedBits {
    // shift out sign, extract exp, subtract bias
    V::SignedBits::from_bits((V::Bits::from_bits(x) << 1) >> 53) - V::SignedBits::splat(0x3FF)
}

#[inline(always)]
fn exponent_f<V: FloatVectorWithBits<Element = f64>>(x: V) -> V {
    let pow2_52: V = crate::const_splat!(f64: 4503599627370496.0);
    let bias: V = crate::const_splat!(f64: 1023.0);

    V::from_bits((V::Bits::from_bits(x) >> 52) | pow2_52.into_bits()) - (pow2_52 + bias)
}

#[inline(always)]
fn ln_d_internal<V: FloatVectorWithBits<Element = f64>, P: Policy, const P1: bool>(x0: V) -> V {
    let ln2_hi = crate::const_splat!(f64: 0.693359375);
    let ln2_lo = crate::const_splat!(f64: -2.121944400546905827679E-4);
    let mut x1 = if P1 { x0 + V::ONE } else { x0 };

    // A subnormal has no exponent field to split, so `fraction2`/`exponent` cannot
    // reduce it and the tail below hands every one of them back as -inf. That is the
    // right answer only because denormals are flushed by default; under `Preserve`,
    // scale them into the normal range by 2^54 and take those 54 powers of two back
    // out of the exponent. The correction then rides the `ln2_hi`/`ln2_lo` multiplies
    // that were happening anyway, so it keeps the full double-word accuracy and costs
    // nothing beyond the scale itself - `ln(5e-324)` is -744.44, not -inf.
    let mut scaled = GenericMask::FALSY;

    if const { matches!(P::POLICY.denormal_behavior, DenormalBehavior::Preserve) } {
        scaled = x1.is_subnormal();
        x1 = x1.mul_c(scaled, crate::const_splat!(f64: hexf::hexf64!("0x1.0p54")));
    }

    let mut x = fraction2::<V>(x1);
    let mut fe = V::cast_from(exponent::<V>(x1));

    let blend = x.cmp_gt(crate::const_splat!(f64: SQRT_2 * 0.5));

    x = x.add_c(!blend, x);
    fe = fe.add_c(blend, V::ONE);

    if const { matches!(P::POLICY.denormal_behavior, DenormalBehavior::Preserve) } {
        fe = fe.sub_c(scaled, crate::const_splat!(f64: 54.0));
    }

    let xp1 = x - V::ONE;

    x = if P1 {
        // log(x+1). Avoid loss of precision when adding 1 and later subtracting 1 if exponent = 0
        fe.cmp_eq(V::ZERO).select(x0, xp1)
    } else {
        // log(x). Expand around 1.0
        xp1
    };

    let x2 = x * x;
    let x3 = x * x2;

    let mut res =
        x3 * x.poly_p::<P, _>(&[
            7.70838733755885391666E0,
            1.79368678507819816313E1,
            1.44989225341610930846E1,
            4.70579119878881725854E0,
            4.97494994976747001425E-1,
            1.01875663804580931796E-4,
        ]) / x.poly_p::<P, _>(&[
            2.31251620126765340583E1,
            7.11544750618563894466E1,
            8.29875266912776603211E1,
            4.52279145837532221105E1,
            1.12873587189167450590E1,
            1.0,
        ]);

    res = fe.mul_adde(ln2_lo, res); // res += fe * ln2_lo;
    res += x2.nmul_adde(V::HALF, x); // res += x - 0.5 * x2;
    res = fe.mul_adde(ln2_hi, res); // res += fe * ln2_hi;

    if const { !P::POLICY.check_overflow } {
        return res;
    }

    let overflow = !x1.is_finite();
    let underflow = x1.cmp_lt(crate::const_splat!(f64: 2.2250738585072014E-308));

    if const { !P::POLICY.avoid_branching } && crate::likely((overflow | underflow).none()) {
        return res;
    }

    res = underflow.select(V::NAN, res); // x1 < 0 gives NAN
    res = x1.is_zero_or_subnormal().select(V::NEG_INFINITY, res); // x1 == 0 gives -INF
    res = overflow.select(x1, res); // INF or NAN goes through
    res = (x1.is_infinite() & x1.is_negative()).select(V::NAN, res); // -INF gives NAN

    res
}

#[inline(always)]
fn atan_internal<V: FloatVectorWithBits<Element = f64>, P: Policy, const ATAN2: bool>(y: V, x: V) -> V {
    let morebits: V = crate::const_splat!(f64: 6.123233995736765886130E-17);
    let morebitso2: V = crate::const_splat!(f64: 6.123233995736765886130E-17 * 0.5);
    let t3po8: V = crate::const_splat!(f64: SQRT_2 + 1.0);

    let mut swapxy = GenericMask::FALSY;

    let t = if ATAN2 {
        let x1 = x.abs().flush_denormals::<P>();
        let y1 = y.abs().flush_denormals::<P>();

        swapxy = y1.cmp_gt(x1);

        let mut x2 = swapxy.select(y1, x1);
        let mut y2 = swapxy.select(x1, y1);

        if const { P::POLICY.check_overflow } {
            let both_inf = x.is_infinite() & y.is_infinite();

            // TODO: Benchmark this branch
            if crate::unlikely(both_inf.any()) {
                x2 = both_inf.select(x2 & V::NEG_ONE, x2);
                y2 = both_inf.select(y2 & V::NEG_ONE, y2);
            }
        }

        y2 / x2
    } else {
        y.abs()
    };

    let t = t.flush_denormals::<P>();

    let not_big = t.cmp_le(t3po8);
    let not_small = t.cmp_ge(crate::const_splat!(f64: 0.66));

    let s = not_big.select(V::FRAC_PI_4, V::FRAC_PI_2);
    let fac = not_big.select(morebitso2, morebits);

    // lightweight select logic using zeroing and conditional adds
    let a = V::NEG_ONE.zz(not_small).add_c(not_big, t);
    let b = V::ONE.zz(not_big).add_c(not_small, t);

    let z = a / b;

    let zz = z * z;

    let re0 = zz.poly_p::<P, _>(&[
        -6.485021904942025371773E1,
        -1.228866684490136173410E2,
        -7.500855792314704667340E1,
        -1.615753718733365076637E1,
        -8.750608600031904122785E-1,
    ]) / zz.poly_p::<P, _>(&[
        1.945506571482613964425E2,
        4.853903996359136964868E2,
        4.328810604912902668951E2,
        1.650270098316988542046E2,
        2.485846490142306297962E1,
        1.0,
    ]);

    // place additions before mul_add to lessen dependency chain
    // also use conditional adds to avoid branching
    let mut re = re0.mul_adde(z * zz, z.add_c(not_small, s).add_c(not_small, fac));

    if ATAN2 {
        re = swapxy.select(V::FRAC_PI_2 - re, re);
        re = (x | y).cmp_eq(V::ZERO).select(V::ZERO, re); // atan2(0,0) = 0 by convention
        // also for x = -0.
        re = x.select_negative(V::PI - re, re);
    }

    re.mul_sign(y)
}

#[inline(always)]
fn asin_internal<V: FloatVectorWithBits<Element = f64>, P: Policy, const ACOS: bool>(x: V) -> V {
    let xa = x.abs().flush_denormals::<P>();

    let is_big = xa.cmp_ge(crate::const_splat!(f64: 0.625));

    let x1 = is_big.select(V::ONE - xa, xa * xa);

    let mut px = V::EMPTY;
    let mut qx = V::EMPTY;
    let mut rx = V::EMPTY;
    let mut sx = V::EMPTY;
    let mut xb = V::EMPTY;

    // if not all are big (if any are small)
    if const { P::POLICY.avoid_branching } || !is_big.all() {
        px = x1.poly_rev_p::<P, _>(&[
            4.253011369004428248960E-3,
            -6.019598008014123785661E-1,
            5.444622390564711410273E0,
            -1.626247967210700244449E1,
            1.956261983317594739197E1,
            -8.198089802484824371615E0,
        ]);

        qx = x1.poly_rev_p::<P, _>(&[
            1.0,
            -1.474091372988853791896E1,
            7.049610280856842141659E1,
            -1.471791292232726029859E2,
            1.395105614657485689735E2,
            -4.918853881490881290097E1,
        ]);
    }

    // if any are big
    if const { P::POLICY.avoid_branching } || is_big.any() {
        xb = (x1 + x1).sqrt();

        rx = x1.poly_p::<P, _>(&[
            2.853665548261061424989E1,
            -2.556901049652824852289E1,
            6.968710824104713396794E0,
            -5.634242780008963776856E-1,
            2.967721961301243206100E-3,
        ]);

        sx = x1.poly_p::<P, _>(&[
            3.424398657913078477438E2,
            -3.838770957603691357202E2,
            1.470656354026814941758E2,
            -2.194779531642920639778E1,
            1.0,
        ]);
    }

    let vx = is_big.select(rx, px);
    let wx = is_big.select(sx, qx);

    let y1 = vx / wx * x1;

    // avoid branching again for this single instruction, just do it
    let z1 = xb.mul_adde(y1, xb);
    let z2 = xa.mul_adde(y1, xa);

    if ACOS {
        let z1 = x.select_negative(V::PI - z1, z1);
        let z2 = V::FRAC_PI_2 - z2.mul_sign(x);
        is_big.select(z1, z2)
    } else {
        let z1 = V::FRAC_PI_2 - z1;
        is_big.select(z1, z2).mul_sign(x)
    }
}

#[inline(always)]
fn pow2n_d<V: FloatVectorWithBits<Element = f64>>(n: V) -> V {
    let pow2_52: V = crate::const_splat!(f64: 4503599627370496.0);
    let bias: V = crate::const_splat!(f64: 1023.0);

    V::from_bits(V::Bits::from_bits(n + (bias + pow2_52)) << 52)
}

/// Split 2^n into two multiplications so neither one leaves normal range, the
/// counterpart of `ps.rs`'s `pow2n_f_safe`. This is what lets the Best-tier exp
/// family cover its entire domain: one `pow2n_d` caps the scale at `2^1023`, which
/// forfeits results in `(2^1023 * 1.42, DBL_MAX]` and every subnormal, while two
/// halves reach both ends exactly.
#[inline(always)]
fn pow2n_d_safe<V: FloatVectorWithBits<Element = f64>>(n: V) -> (V, V) {
    // Split n into two halves, each comfortably inside the exponent range
    let half = n.scale(0.5).floor();
    let other = n - half;

    (pow2n_d(half), pow2n_d(other))
}

#[inline(always)]
fn exp_d_internal<V: FloatVectorWithBits<Element = f64>, P: Policy, const MODE: u8>(x0: V) -> V {
    let mut x = x0.flush_denormals::<P>();
    let mut r;

    let max_x;

    match MODE {
        EXP_MODE_POW2 | EXP_MODE_POW2M1 => {
            max_x = 1022.0;

            r = x.round();

            x -= r;
            x *= V::LN_2;
        }
        EXP_MODE_POW10 | EXP_MODE_POW10M1 => {
            max_x = 307.65;

            let log10_2_hi: V = crate::const_splat!(f64: -0.30102999554947019); // log10(2) in two parts
            let log10_2_lo: V = crate::const_splat!(f64: -1.1451100899212592E-10);

            r = (x * crate::const_splat!(f64: LN_10 * LOG2_E)).round();

            x = r.mul_adde(log10_2_hi, x); // x -= r * log10_2_hi;
            x = r.mul_adde(log10_2_lo, x); // x -= r * log10_2_lo;
            x *= V::LN_10;
        }
        _ => {
            // The single-`pow2n_d` bound: `round(x * log2 e)` (minus one for EXPH)
            // must stay <= 1023, and `(1 + z) * 2^1023` tops out at ~1.42 * 2^1023,
            // safely below DBL_MAX. The Best tier replaces these bounds with the
            // true per-mode domain below, using the two-part `pow2n_d_safe`.
            max_x = const {
                match MODE {
                    EXP_MODE_EXPH => 710.11, // round(x log2 e) <= 1024, minus one after the shift
                    _ => 709.42,             // EXP / EXPM1: round(x log2 e) <= 1023
                }
            };

            let ln2d_hi: V = crate::const_splat!(f64: -0.693145751953125);
            let ln2d_lo: V = crate::const_splat!(f64: -1.42860682030941723212E-6);

            r = (x * crate::const_splat!(f64: LOG2_E)).round();

            x = r.mul_adde(ln2d_hi, x); // x -= r * ln2_hi;
            x = r.mul_adde(ln2d_lo, x); // x -= r * ln2_lo;

            if MODE == EXP_MODE_EXPH {
                r -= V::ONE;
            }
        }
    }

    // Taylor coefficients, 1/n!
    // Not using minimax approximation because we prioritize precision close to x = 0
    let mut z = x.poly_p::<P, _>(&[
        0.0,
        1.0,
        1.0 / 2.0,
        1.0 / 6.0,
        1.0 / 24.0,
        1.0 / 120.0,
        1.0 / 720.0,
        1.0 / 5040.0,
        1.0 / 40320.0,
        1.0 / 362880.0,
        1.0 / 3628800.0,
        1.0 / 39916800.0,
        1.0 / 479001600.0,
        1.0 / 6227020800.0,
    ]);

    z = if const { P::POLICY.precision.gt(PrecisionPolicy::Average) } {
        // Two-part scaling reaches the full domain: results past `1.42 * 2^1023`
        // (which one `pow2n_d` cannot form) and the whole subnormal range, matching
        // the f32 Best path. The extra multiply is nothing next to this tier's
        // polynomial.
        let (n2a, n2b) = pow2n_d_safe::<V>(r);

        match MODE {
            EXP_MODE_EXPM1 | EXP_MODE_POW2M1 | EXP_MODE_POW10M1 => {
                z.mul_adde(n2a, n2a - V::ONE).mul_adde(n2b, n2b - V::ONE)
            }
            _ => z.mul_adde(n2a, n2a) * n2b, // (z + 1) * n2a * n2b
        }
    } else {
        // `pow2n_d` builds the exponent field with an integer add, so an `r` outside
        // `[-1023, 1023]` carries into the _sign_ bit and the result wraps to a negative
        // number rather than saturating: `exp(800)` returned -8.436e-270, `exph(-709)`
        // returned -9.8e307, and `exp_m1(-709.5)` garbage instead of -1.
        //
        // The low side wraps _inside_ the range gate (EXPH's `r - 1` and EXPM1's wider
        // gate both reach -1024), so it is saturated unconditionally. `r = -1023` is the
        // all-zero exponent field, i.e. +0, which makes every underflow land on exactly
        // zero, and on exactly -1 for the M1 modes, since `z * 0 + (0 - 1)` is -1
        // whatever `z` holds.
        //
        // The high side only wraps past the gate, whose select repairs it under
        // `check_overflow`. The clamp stays for the policies that turn that off
        // (`UltraPerformance`, `HighPerformance`). Deliberately 1023 and not 1024,
        // because the all-ones field would be infinity and the recombination is
        // `z * n2 + n2`, where `z` is exactly zero whenever the reduced argument is
        // (every integer input to `exp2`, for one), so `0 * inf` would hand back NaN for
        // the cleanest inputs in the range. Saturating one exponent lower keeps
        // everything finite, the same contract the f32 path has always had.
        let mut rc = r.max(crate::const_splat!(f64: -1023.0));

        if const { !P::POLICY.check_overflow } {
            rc = rc.min(crate::const_splat!(f64: 1023.0));
        }

        let n2 = pow2n_d::<V>(rc);

        match MODE {
            EXP_MODE_EXPM1 | EXP_MODE_POW2M1 | EXP_MODE_POW10M1 => z.mul_adde(n2, n2 - V::ONE),
            _ => z.mul_adde(n2, n2), // (z + 1.0f) * n2
        }
    };

    if const { P::POLICY.check_overflow } {
        let mut in_range = x0.is_finite();

        if const { P::POLICY.precision.gt(PrecisionPolicy::Average) } {
            // With `pow2n_d_safe` the kernel is exact over the whole domain, so the
            // gate can sit at the true per-mode bounds (as the f32 path does): the
            // high end is where the result overflows the format, the low end where
            // it underflows to zero (or saturates at -1 for the M1 modes).
            #[rustfmt::skip]
            let (min_x, max_x) = const { match MODE {
                EXP_MODE_EXP => (-745.2, 709.78),      // (ln(2^-1075), ln(DBL_MAX))
                EXP_MODE_EXPM1 => (-708.39, 709.78),   // (below: exactly -1, ln(DBL_MAX))
                EXP_MODE_EXPH => (-744.5, 710.47),     // EXP shifted by ln 2 either side
                EXP_MODE_POW2 => (-1075.0, 1024.0),    // (2^-1075 rounds to 0, log2(DBL_MAX))
                EXP_MODE_POW2M1 => (-1075.0, 1024.0),
                EXP_MODE_POW10 => (-323.6, 308.25),    // (log10(2^-1075), log10(DBL_MAX))
                EXP_MODE_POW10M1 => (-323.6, 308.25),

                _ => panic!("Invalid MODE for exp_d_internal"),
            }};

            in_range &= x0.cmp_ge(V::splat(min_x)) & x0.cmp_le(V::splat(max_x));
        } else {
            in_range &= x0.abs().cmp_lt(V::splat(max_x));
        }

        if crate::likely(in_range.all()) {
            return z;
        }

        let underflow_value = const {
            if MODE == EXP_MODE_EXPM1 || MODE == EXP_MODE_POW2M1 || MODE == EXP_MODE_POW10M1 {
                V::NEG_ONE
            } else {
                V::ZERO
            }
        };

        r = x0.select_negative(underflow_value, V::INFINITY);
        z = in_range.select(z, r);
        z = x0.is_nan().select(x0, z);
    }

    z
}

/// Cody-Waite range reduction for double-precision trig, the counterpart to `ps.rs`'s.
///
/// Reduces `xa` (absolute value, flushed) modulo pi/2, returning `(x_hi, x_lo, quadrant)`.
/// `x_lo` is always zero here: unlike f32 there is no Payne-Hanek fallback, so beyond
/// the limit below the argument is zeroed rather than reduced. Three-part pi/2 is
/// enough that the arguments this gives up on are rare enough not to pay for on every
/// call.
///
/// When `PI` is true this performs the sinpi/cospi reduction instead - the argument is
/// reduced in units of one half turn and scaled by pi afterwards, which needs no
/// extended-precision split at all.
#[thermite_macros::dispatch(V, thermite = "crate")]
fn payne_hanek_reduction<P: Policy, V: FloatVectorWithBits<Element = f64>>(xa: &V) -> (V, V, V::Bits) {
    let xa_bits: V::Bits = xa.into_bits();

    // Extract unbiased exponent and significand
    let exp =
        (V::SignedBits::from_bits(xa_bits.shri::<52>()) & V::SignedBits::splat(0x7FF)) - V::SignedBits::splat(1023);
    let exp_u: V::Unsigned = V::Bits::from_bits(exp.max(V::SignedBits::ZERO)).cast();

    // 53-bit significand with implicit hidden bit restored
    let sig = (xa_bits & V::Bits::splat(0x000F_FFFF_FFFF_FFFF)) | V::Bits::splat(0x0010_0000_0000_0000);

    // Padded 2/pi table: one zero word prepended to absorb the -55 offset.
    // Index with (exp + 9) instead of (exp - 55) to avoid unsigned underflow.
    // 18 fraction words = 1152 bits, enough for the max f64 exponent (1023):
    // window start (exp - 55) + 128 window bits <= 1096 < 1152.
    const INVPI_TABLE: [u64; 19] = [
        0x0000000000000000, // padding
        0xA2F9836E4E441529,
        0xFC2757D1F534DDC0,
        0xDB6295993C439041,
        0xFE5163ABDEBBC561,
        0xB7246E3A424DD2E0,
        0x06492EEA09D1921C,
        0xFE1DEB1CB129A73E,
        0xE88235F52EBB4484,
        0xE99C7026B45F7E41,
        0x3991D639835339F4,
        0x9C845F8BBDF9283B,
        0x1FF897FFDE05980F,
        0xEF2F118B5A0A6D1F,
        0x6D367ECF27CB09B7,
        0x4F463F669E5FEA2D,
        0x7527BAC7EBE5F17B,
        0x3D0739F78A5292EA,
        0x6BFB5FB11F8D5D08,
    ];

    let biased = exp_u + V::Unsigned::splat(9); // always >= 9, never underflows
    let idx: V::Unsigned = biased.shri::<6>();
    let shift = biased & V::Unsigned::splat(63);
    let inv_shift = (V::Unsigned::splat(64) - shift) & V::Unsigned::splat(63);

    let c0 = unsafe { V::Unsigned::lookup_unchecked(&INVPI_TABLE, idx) };
    let c1 = unsafe { V::Unsigned::lookup_unchecked(&INVPI_TABLE, idx + V::Unsigned::ONE) };
    let c2 = unsafe { V::Unsigned::lookup_unchecked(&INVPI_TABLE, idx + V::Unsigned::TWO) };

    // Shift chunks to align binary point.
    // Mask shifts by 63 to prevent UB on shift == 64 in some ISAs.
    let mask = shift.cmp_ne(V::Unsigned::ZERO);
    let aligned_hi = c0.shlv(shift) | c1.shrv(inv_shift).zz(mask);
    let aligned_lo = c1.shlv(shift) | c2.shrv(inv_shift).zz(mask);

    let aligned_hi: V::Bits = aligned_hi.cast();
    let aligned_lo: V::Bits = aligned_lo.cast();

    // Multiply significand by aligned chunks.
    // 181-bit product: sig(53) * aligned(128); only the low 128 bits matter,
    // everything above integer bit 127 is a multiple of 4 (discarded mod 4).
    // Binary point at bit 125: bits 126:125 = quadrant, bits 124:0 = fraction.
    let prod_hi = sig.mullo(aligned_hi); // bits 127:64 (low half of sig * hi)
    let prod_lo = sig.mulhi(aligned_lo); // bits 116:64 (high half of sig * lo)
    let mid_bits = prod_hi + prod_lo; // bits 127:64 of the product
    let prod_lo_lo = sig.mullo(aligned_lo); // bits 63:0

    // Extract quadrant from bits 62:61 of mid_bits.
    let mut q_ph: V::Bits = mid_bits.shri::<61>() & V::Bits::splat(3);

    // 61-bit fraction: mid_bits bits 60:0, extended by prod_lo_lo below.
    let fraction_hi_int = mid_bits & V::Bits::splat(0x1FFF_FFFF_FFFF_FFFF);

    // Reconstruct as double-double (two non-overlapping f64 values).
    //
    // frac_hi: top 52 bits of fraction_hi_int, injected as f64 mantissa (exact).
    // Represents (fraction_hi_int >> 9) * 2^-52.
    let frac_hi_bits = fraction_hi_int.shri::<9>() | V::Bits::splat(0x3FF0_0000_0000_0000);
    let frac_hi = V::from_bits(frac_hi_bits) - V::ONE;

    // frac_lo: bottom 9 bits of fraction_hi_int | top 43 bits of prod_lo_lo = 52 bits.
    // Represents residual * 2^-104. Exact since residual <= 2^52 - 1.
    let residual = (fraction_hi_int & V::Bits::splat(0x1FF)).shli::<43>() | prod_lo_lo.shri::<21>();
    let frac_lo_int: V::SignedBits = residual.cast();
    let frac_lo = V::cast_from(frac_lo_int) * crate::const_splat!(f64: hexf::hexf64!("0x1.0p-104"));

    // Center from [0, 1) to [-0.5, 0.5) to match Cody-Waite's round().
    // Only frac_hi needs adjustment; frac_lo is unchanged since
    // (frac_hi - 1) + frac_lo = old_total - 1.
    let needs_round = frac_hi.cmp_ge(V::HALF);
    let frac_hi = frac_hi.sub_c(needs_round, V::ONE);
    q_ph = q_ph.add_c(needs_round.cast(), V::Bits::ONE);

    // Multiply by π/2 as double-double.
    // π/2 = pi2_hi + pi2_lo where pi2_hi = f64(π/2) and pi2_lo = π/2 - f64(π/2).
    let pi2_hi = V::FRAC_PI_2;
    let pi2_lo = crate::const_splat!(f64: 6.123233995736766e-17);

    let x_hi = frac_hi * pi2_hi;

    // Recover rounding error via exact FMA, then add cross terms.
    // frac_lo * pi2_lo is O(2^-158), negligible.
    let x_lo = frac_hi.mul_add(pi2_hi, -x_hi) + frac_hi * pi2_lo + frac_lo * pi2_hi;

    (x_hi, x_lo, q_ph)
}

#[inline(always)]
pub(crate) fn trig_range_reduction<P: Policy, V: FloatVectorWithBits<Element = f64>, const PI: bool>(
    mut xa: V,
) -> (V, V, V::Bits) {
    let mut is_large = V::Mask::FALSY;

    let y = if PI {
        xa + xa // 2x for sinpi/cospi
    } else {
        // Without true FMA, `y * dp1` (30-bit dp1) is only exact while y fits in
        // 23 bits, i.e. |x| <~ 1.3e7 - beyond that Cody-Waite quietly loses bits.
        // At Best+ (where Payne-Hanek takes over) hand off there; at <= Average
        // the limit only gates the bounded clamp, so keep the old wider window.
        is_large = xa.cmp_gt(if const { P::POLICY.precision.gt(PrecisionPolicy::Average) } {
            crate::const_splat!(<V> = <V: FloatVectorWithBits> f64: {
                match V::HAS_TRUE_FMA {
                    true => 1e15,
                    false => 1e7,
                }
            })
        } else {
            crate::const_splat!(<V> = <V: FloatVectorWithBits> f64: {
                match V::HAS_TRUE_FMA {
                    true => 1e15,
                    false => 1e13,
                }
            })
        });

        // At Average precision and below there is no Payne-Hanek fallback (it is
        // reserved for Best+; these tiers stay fast): zero out-of-range lanes so
        // they at least produce a bounded result.
        if const { P::POLICY.check_overflow && P::POLICY.precision.le(PrecisionPolicy::Average) } {
            xa = xa.nz(is_large); // set to zero if too large
        }

        xa.scale(FloatConsts::FRAC_2_PI)
    };

    let y = y.round();

    let mut q = V::Bits::fast_cast_from(y);

    // pi/2 split into three parts for extended precision modular arithmetic
    let dp1 = crate::const_splat!(f64: 7.853981554508209228515625E-1 * 2.0);
    let dp2 = crate::const_splat!(f64: 7.94662735614792836714E-9 * 2.0);
    let dp3 = crate::const_splat!(f64: 3.06161699786838294307E-17 * 2.0);

    // Reduce by extended precision modular arithmetic
    // x = ((xa - y * DP1) - y * DP2) - y * DP3;
    // or if calculating sinpi/cospi:
    // x = pi * (xa - y * 0.5)
    let mut x = if PI {
        y.nmul_adde(V::HALF, xa).scale(FloatConsts::PI)
    } else if const { V::HAS_TRUE_FMA } {
        // if true FMA is available, we only have to do two FMAs
        y.nmul_add(dp3, y.nmul_add(dp2 + dp1, xa))
    } else {
        ((xa - y * dp1) - y * dp2) - y * dp3
    };

    let mut x_lo = V::ZERO;

    // Payne-Hanek fallback for large arguments (non-PI only, Best+ precision).
    // Non-finite lanes are excluded so inf/NaN still propagate NaN via Cody-Waite.
    if const { P::POLICY.precision.gt(PrecisionPolicy::Average) && !PI }
        && (P::POLICY.avoid_branching || is_large.any())
    {
        let is_large = is_large & xa.is_finite();

        let (x_ph, x_lo_ph, q_ph) = payne_hanek_reduction::<P, V>(&xa);

        x = is_large.select(x_ph, x);
        x_lo = x_lo_ph.zz(is_large); // zero out x_lo when not using Payne-Hanek
        q = is_large.select(q_ph, q);
    }

    (x, x_lo, q)
}

#[inline(always)]
fn sincos_d_internal<P: Policy, V: FloatVectorWithBits<Element = f64>, const PI: bool>(xx: V) -> (V, V) {
    let xa = xx.abs().flush_denormals::<P>();

    let (x, x_lo, q) = trig_range_reduction::<P, V, PI>(xa);

    // Taylor expansion of sin and cos, valid for -pi/4 <= x <= pi/4
    let x2 = x * x;
    let x4 = x2 * x2;

    let mut s = x2.poly_rev_p::<P, _>(&[
        1.58962301576546568060E-10,
        -2.50507477628578072866E-8,
        2.75573136213857245213E-6,
        -1.98412698295895385996E-4,
        8.33333333332211858878E-3,
        -1.66666666666666307295E-1,
    ]);

    let mut c = x2.poly_rev_p::<P, _>(&[
        -1.13585365213876817300E-11,
        2.08757008419747316778E-9,
        -2.75573141792967388112E-7,
        2.48015872888517045348E-5,
        -1.38888888888730564116E-3,
        4.16666666666665929218E-2,
    ]);

    let mut x0 = x;

    if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
        x0 += x_lo; // fold in the Payne-Hanek low word for Best+ precision
    }

    s = s.mul_adde(x2 * x, x0); // s = x + (x * x2) * s;
    c = c.mul_adde(x4, x2.nmul_adde(V::HALF, V::ONE)); // c = 1.0 - x2 * 0.5 + (x2 * x2) * c;

    if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
        c = x.nmul_adde(x_lo, c); // d cos = -sin ~= -x for the low word
    }

    // swap sin and cos if odd quadrant
    let swap = (q & V::Bits::ONE).cmp_ne(V::Bits::ZERO);

    if const { P::POLICY.check_overflow } {
        // `q` is the exact integer form of the quotient the reduction rounded, and the
        // quotient is non-negative because `xa` is an absolute value, so testing it is
        // equivalent to the old test on that float. `xa` here is pre-clamp, which also
        // agrees: the clamp only fires above the limit, and there the quotient is zero.
        let overflow = q.cmp_gt(V::Bits::splat((1u64 << 52) - 1)).cast::<V::Mask>() & xa.is_finite();

        s = s.nz(overflow); // overflow.select(V::ZERO, s);
        c = overflow.select(V::ONE, c);
    }

    let sin1 = swap.select(c, s);
    let cos1 = swap.select(s, c);

    let signsin = V::from_bits(q << 62) ^ xx;
    let signcos = V::from_bits(((q + V::Bits::ONE) & V::Bits::splat(2)) << 62);

    // combine signs
    (sin1.mul_sign(signsin), cos1 ^ signcos)
}
