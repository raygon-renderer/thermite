use crate::divider::Divider;
use core::f64::consts::{LN_10, LOG2_E, SQRT_2};

use super::*;

impl<V: FloatVectorWithBits<Element = f64>> SpecializedCoreMath<f64> for V {
    #[inline(always)]
    fn inverse_sqrt<P: Policy>(self) -> Self {
        super::generic::inverse_sqrt_internal::<V, f64, P>(self)
    }
}

impl<V: FloatVectorWithBits<Element = f64>> SpecializedRealMath<f64> for V {
    #[inline(always)]
    fn atan2<P: Policy>(self, x: Self) -> Self {
        atan_internal::<Self, P, true>(self, x)
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
        let x = x0.abs();
        let y = x.exph_p::<P>();
        let qy = V::splat(0.25) / y;

        let mut sinh = y - qy;
        let cosh = y + qy;

        let x_small = x.cmp_le(V::ONE);

        // if any are small, use a polynomial approximation
        if P::POLICY.avoid_branching || x_small.any() {
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
        let x = x0.abs();

        let x_small = x.cmp_le(V::ONE);

        let mut y2 = V::EMPTY;

        if P::POLICY.avoid_branching || !x_small.all() {
            y2 = x.exph_p::<P>();
            y2 -= V::splat(0.25) / y2;

            // if we don't care about small x, we can skip the next branch
            if const { P::POLICY.avoid_precision_branches() } {
                return y2.mul_sign(x0);
            }
        }

        if P::POLICY.avoid_branching || x_small.any() {
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
        y + V::splat(0.25) / y
    }

    #[inline(always)]
    fn tanh<P: Policy>(self) -> Self {
        let x0 = self;
        let x = x0.abs();

        let x_small = x.cmp_le(V::splat(0.625));

        let mut y2 = V::EMPTY;

        if P::POLICY.avoid_branching || !x_small.all() {
            y2 = (x + x).exp_p::<P>();
            y2 = (y2 - V::ONE) / (y2 + V::ONE); // originally (1 - 2/(y2 + 1))

            if P::POLICY.check_overflow {
                y2 = x.cmp_gt(V::splat(350.0)).select(V::ONE, y2);
            }

            if const { P::POLICY.avoid_precision_branches() } {
                return y2.mul_sign(x0);
            }
        }

        if P::POLICY.avoid_branching || x_small.any() {
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
        let x = x0.abs();
        let x2 = x * x;

        let x_small = x.cmp_le(V::splat(0.533));

        let mut y2 = V::EMPTY;

        if P::POLICY.avoid_branching || !x_small.all() {
            y2 = ((x2 + V::ONE).sqrt() + x).ln_p::<P>();

            if const { P::POLICY.check_overflow || !P::POLICY.avoid_precision_branches() } {
                let x_huge = x.cmp_gt(V::splat(1e20));

                if crate::unlikely(x_huge.any()) {
                    y2 = x_huge.select(x.ln_p::<P>() + V::LN_2, y2);
                }
            }
        }

        if P::POLICY.avoid_branching || x_small.any() {
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
        let x0 = self;
        let x1 = x0 - V::ONE;

        let x_small = x1.cmp_le(V::splat(0.49));

        let mut y2 = V::EMPTY;

        if P::POLICY.avoid_branching || !x_small.all() {
            y2 = (x0.mul_sube(x0, V::ONE).sqrt() + x0).ln_p::<P>();

            if const { P::POLICY.check_overflow && !P::POLICY.avoid_precision_branches() } {
                let x_huge = x1.cmp_gt(V::splat(1e20));

                if crate::unlikely(x_huge.any()) {
                    y2 = x_huge.select(x0.ln_p::<P>() + V::LN_2, y2);
                }
            }

            if const { P::POLICY.avoid_precision_branches() } {
                // certain overflow checks can still be important even if precision is not
                if P::POLICY.check_overflow {
                    y2 = x0.cmp_lt(V::ONE).select(V::NAN, y2);
                }

                return y2;
            }
        }

        if P::POLICY.avoid_branching || x_small.any() {
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

            if P::POLICY.check_overflow {
                y1 = x0.cmp_lt(V::ONE).select(V::NAN, y1);
            }

            y2 = x_small.select(y1, y2);
        }

        y2
    }

    #[inline(always)]
    fn atanh<P: Policy>(self) -> Self {
        let x0 = self;
        let x = x0.abs();

        let x_small = x.cmp_le(V::HALF);

        let mut y2 = V::EMPTY;

        if P::POLICY.avoid_branching || !x_small.all() {
            y2 = ((V::ONE + x) / (V::ONE - x)).ln_p::<P>() * V::HALF;

            if P::POLICY.check_overflow {
                let y3 = x.cmp_eq(V::ONE).select(V::INFINITY, V::NAN);
                y2 = x.cmp_ge(V::ONE).select(y3, y2);
            }

            if const { P::POLICY.avoid_precision_branches() } {
                return y2.mul_sign(x0);
            }
        }

        if P::POLICY.avoid_branching || x_small.any() {
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
    fn powf<P: Policy>(self, y: Self) -> Self {
        let x0 = self;

        // define constants
        let ln2d_hi = crate::generic_splat!(f64: 0.693145751953125); // log(2) in extra precision, high bits
        let ln2d_lo = crate::generic_splat!(f64: 1.42860682030941723212E-6); // low bits of log(2)

        let x1 = x0.abs();

        let mut x = fraction2(x1);

        let blend = x.cmp_gt(V::splat(SQRT_2 / 2.0));

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
        let e2 = (lg * y * V::LOG2_E).round();

        // subtract this from lg, with extra precision
        let mut v = e2.nmul_adde(ln2d_lo, lg.mul_sube(y, e2 * ln2d_hi));

        // add remainder from ef * y
        v = yr.mul_adde(V::LN_2, v); // v += yr * VM_LN2;

        // correct for previous rounding errors
        v = (lgerr + x2err).nmul_adde(y, v); // v -= (lgerr + x2err) * y;

        // extract something for the exponent if possible
        let mut x = v;
        let e3 = (x * V::LOG2_E).round();

        // high precision multiplication not needed here because abs(e3) <= 1
        x = e3.nmul_adde(V::LN_2, x); // x -= e3 * VM_LN2;

        // Taylor coefficients for exp function, 1/n!
        let z = x.poly_p::<P, _>(&[
            1.0, // + 1
            1.0, // 1x
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

        // contributions to exponent
        let ee = e1 + e2 + e3;
        let ei: V::SignedBits = ee.fast_cast();

        // biased exponent of result:
        let ej = ei + (V::SignedBits::from_bits(x.abs()) >> 52);

        // add exponent by signed integer addition
        let mut z = V::from_bits(V::SignedBits::from_bits(z) + (ei << 52));

        if !P::POLICY.check_overflow {
            return z;
        }

        // check exponent for overflow and underflow
        let overflow = ej.cmp_ge(V::SignedBits::splat(0x07FF)).cast::<V::Mask>() | ee.cmp_gt(V::splat(3000.0));
        let underflow = ej.cmp_le(V::SignedBits::splat(0x0000)).cast::<V::Mask>() | ee.cmp_lt(V::splat(-3000.0));

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

        let not_special = (xfinite & yfinite & (efinite | xzero));

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
                    yodd & z,               // 0.0 with the sign of z from above
                    x0.abs() | (x0 & yodd), // get sign of x0 only if y is odd integer
                ),
            ),
        );

        // Always propagate nan:
        // Deliberately differing from the IEEE-754 standard which has pow(0,nan)=1, and pow(1,nan)=1
        (x0.is_nan() | y.is_nan()).select(x0 + y, z1)
    }

    #[inline(always)]
    fn cbrt<P: Policy>(self) -> Self {
        let x = self;

        let b1 = crate::generic_splat!(u64: 715094163); // B1 = (1023-1023/3-0.03306235651)*2**20
        let b2 = crate::generic_splat!(u64: 696219795); // B2 = (1023-1023/3-54/3-0.03306235651)*2**20
        let m = crate::generic_splat!(u64: 0x7fffffff); // u32::MAX >> 1

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

        let r = if const { P::POLICY.precision.ge(PrecisionPolicy::Best) || !Self::HAS_TRUE_FMA } {
            // original form, 5 simple ops, 2 divisions
            let xtt = x / (t * t);
            (xtt - t) / ((t + t) + xtt)
        } else {
            // fast form, 3 simple ops, 1 division, 1 fma
            let t3 = t * t * t;
            (x - t3) / t3.mul_add(Self::TWO, x)
        };

        t = r.mul_adde(t, t);

        if !P::POLICY.check_overflow {
            return x.cmp_eq(Self::ZERO).select(x, t);
        }

        (hx0.cmp_gt(V::Bits::splat(0x7f800000)) | hx0.cmp_eq(V::Bits::ZERO)).select(x, t)
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
        ln_d_internal::<Self, P, false>(self) * V::LOG2_E
    }

    #[inline(always)]
    fn log10<P: Policy>(self) -> Self {
        ln_d_internal::<Self, P, false>(self) * V::LOG10_2
    }

    #[inline(always)]
    fn ln1m_expnx<P: Policy>(self) -> Self {
        (V::ONE - (-self).exp_p::<P>()).ln_p::<P>()
    }

    #[inline(always)]
    fn ln1m_expnx_ext<P: Policy>(self, _lnx: Self) -> Self {
        (V::ONE - (-self).exp_p::<P>()).ln_p::<P>()
    }
}

#[inline(always)]
fn fraction2<V: FloatVectorWithBits<Element = f64>>(x: V) -> V {
    // set exponent to 0 + bias
    (x & V::splat(f64::from_bits(0x000FFFFFFFFFFFFF))) | V::splat(f64::from_bits(0x3FE0000000000000))
}

#[inline(always)]
fn exponent<V: FloatVectorWithBits<Element = f64>>(x: V) -> V::SignedBits {
    // shift out sign, extract exp, subtract bias
    V::SignedBits::from_bits((V::Bits::from_bits(x) << 1) >> 53) - V::SignedBits::splat(0x3FF)
}

#[inline(always)]
fn exponent_f<V: FloatVectorWithBits<Element = f64>>(x: V) -> V {
    let pow2_52: V = crate::generic_splat!(f64: 4503599627370496.0);
    let bias: V = crate::generic_splat!(f64: 1023.0);

    V::from_bits((V::Bits::from_bits(x) >> 52) | pow2_52.into_bits()) - (pow2_52 + bias)
}

#[inline(always)]
fn ln_d_internal<V: FloatVectorWithBits<Element = f64>, P: Policy, const P1: bool>(x0: V) -> V {
    let ln2_hi = crate::generic_splat!(f64: 0.693359375);
    let ln2_lo = crate::generic_splat!(f64: -2.121944400546905827679E-4);
    let x1 = if P1 { x0 + V::ONE } else { x0 };

    let mut x = fraction2::<V>(x1);
    let mut fe = V::cast_from(exponent::<V>(x1));

    let blend = x.cmp_gt(V::splat(SQRT_2 * 0.5));

    x = blend.select(x, x + x); // x = x.conditional_add(x, !blend);
    fe = blend.select(fe + V::ONE, fe); // fe = fe.conditional_add(V::ONE, blend);

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

    if !P::POLICY.check_overflow {
        return res;
    }

    let overflow = !x1.is_finite();
    let underflow = x1.cmp_lt(V::splat(2.2250738585072014E-308));

    if !P::POLICY.avoid_branching && crate::likely((overflow | underflow).none()) {
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
    let morebits = V::splat(6.123233995736765886130E-17);
    let morebitso2 = V::splat(6.123233995736765886130E-17 * 0.5);
    let t3po8 = V::splat(SQRT_2 + 1.0);

    let mut swapxy = GenericMask::FALSY;

    let t = if ATAN2 {
        let x1 = x.abs();
        let y1 = y.abs();

        swapxy = y1.cmp_gt(x1);

        let mut x2 = swapxy.select(y1, x1);
        let mut y2 = swapxy.select(x1, y1);

        if P::POLICY.check_overflow {
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

    let not_big = t.cmp_le(t3po8);
    let not_small = t.cmp_ge(V::splat(0.66));

    let s = not_big.select(V::FRAC_PI_4, V::FRAC_PI_2);
    let fac = not_big.select(morebitso2, morebits);

    // lightweight select logic using zeroing and conditional adds
    let a = V::NEG_ONE.z(not_small).add_c(not_big, t);
    let b = V::ONE.z(not_big).add_c(not_small, t);

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
    let xa = x.abs();

    let is_big = xa.cmp_ge(V::splat(0.625));

    let x1 = is_big.select(V::ONE - xa, xa * xa);

    let mut px = V::EMPTY;
    let mut qx = V::EMPTY;
    let mut rx = V::EMPTY;
    let mut sx = V::EMPTY;
    let mut xb = V::EMPTY;

    // if not all are big (if any are small)
    if P::POLICY.avoid_branching || !is_big.all() {
        px = x1.poly_p::<P, _>(&[
            -8.198089802484824371615E0,
            1.956261983317594739197E1,
            -1.626247967210700244449E1,
            5.444622390564711410273E0,
            -6.019598008014123785661E-1,
            4.253011369004428248960E-3,
        ]);

        qx = x1.poly_p::<P, _>(&[
            -4.918853881490881290097E1,
            1.395105614657485689735E2,
            -1.471791292232726029859E2,
            7.049610280856842141659E1,
            -1.474091372988853791896E1,
            1.0,
        ]);
    }

    // if any are big
    if P::POLICY.avoid_branching || is_big.any() {
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
    let pow2_52: V = crate::generic_splat!(f64: 4503599627370496.0);
    let bias: V = crate::generic_splat!(f64: 1023.0);

    V::from_bits(V::Bits::from_bits(n + (bias + pow2_52)) << 52)
}

#[inline(always)]
fn exp_d_internal<V: FloatVectorWithBits<Element = f64>, P: Policy, const MODE: u8>(x0: V) -> V {
    let mut x = x0;
    let mut r;

    let max_x;

    match MODE {
        EXP_MODE_POW2 => {
            max_x = 1022.0;

            r = x0.round();

            x -= r;
            x *= V::LN_2;
        }
        EXP_MODE_POW10 => {
            max_x = 307.65;

            let log10_2_hi = V::splat(0.30102999554947019); // log10(2) in two parts
            let log10_2_lo = V::splat(1.1451100899212592E-10);

            r = (x0 * V::splat(LN_10 * LOG2_E)).round();

            x = r.nmul_adde(log10_2_hi, x); // x -= r * log10_2_hi;
            x = r.nmul_adde(log10_2_lo, x); // x -= r * log10_2_lo;
            x *= V::LN_10;
        }
        _ => {
            max_x = const { if MODE == EXP_MODE_EXP { 708.39 } else { 709.7 } };

            let ln2d_hi = V::splat(0.693145751953125);
            let ln2d_lo = V::splat(1.42860682030941723212E-6);

            r = (x0 * V::splat(LOG2_E)).round();

            x = r.nmul_adde(ln2d_hi, x); // x -= r * ln2_hi;
            x = r.nmul_adde(ln2d_lo, x); // x -= r * ln2_lo;

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

    let n2 = pow2n_d::<V>(r);

    z = match MODE {
        EXP_MODE_EXPM1 => z.mul_adde(n2, n2 - V::ONE),
        _ => z.mul_adde(n2, n2), // (z + 1.0f) * n2
    };

    if P::POLICY.check_overflow {
        let in_range = x0.abs().cmp_lt(V::splat(max_x)) & x0.is_finite();

        if crate::likely(in_range.all()) {
            return z;
        }

        let underflow_value = const { if MODE == EXP_MODE_EXPM1 { V::NEG_ONE } else { V::ZERO } };

        r = x0.select_negative(underflow_value, V::INFINITY);
        z = in_range.select(z, r);
        z = x0.is_nan().select(x0, z);
    }

    z
}

#[inline(always)]
fn sincos_d_internal<P: Policy, V: FloatVectorWithBits<Element = f64>, const PI: bool>(xx: V) -> (V, V) {
    let mut xa = xx.abs();

    let y = if PI {
        xa + xa // 2x for sinpi/cospi
    } else {
        if const { P::POLICY.check_overflow } {
            let limit: V = crate::generic_splat!(<V> = <V: FloatVectorWithBits> f64: {
                match V::HAS_TRUE_FMA {
                    true => 1e15,
                    false => 1e13,
                }
            });

            xa = xa.z(xa.cmp_le(limit)); // set to zero if too large
        }

        xa * V::FRAC_2_PI
    };

    let y = y.round();

    let q = V::Bits::fast_cast_from(y);

    // pi/2 split into three parts for extended precision modular arithmetic
    let dp1 = crate::generic_splat!(f64: 7.853981554508209228515625E-1 * 2.0);
    let dp2 = crate::generic_splat!(f64: 7.94662735614792836714E-9 * 2.0);
    let dp3 = crate::generic_splat!(f64: 3.06161699786838294307E-17 * 2.0);

    // Reduce by extended precision modular arithmetic
    // x = ((xa - y * DP1) - y * DP2) - y * DP3;
    // or if calculating sinpi/cospi:
    // x = pi * (xa - y * 0.5)
    let x = if PI {
        y.nmul_adde(V::HALF, xa) * V::PI
    } else if const { V::HAS_TRUE_FMA } {
        // if true FMA is available, we only have to do two FMAs
        y.nmul_add(dp3, y.nmul_add(dp2 + dp1, xa))
    } else {
        ((xa - y * dp1) - y * dp2) - y * dp3
    };

    // Taylor expansion of sin and cos, valid for -pi/4 <= x <= pi/4
    let x2 = x * x;
    let x4 = x2 * x2;

    let mut s = x2.poly_p::<P, _>(&[
        -1.66666666666666307295E-1,
        8.33333333332211858878E-3,
        -1.98412698295895385996E-4,
        2.75573136213857245213E-6,
        -2.50507477628578072866E-8,
        1.58962301576546568060E-10,
    ]);

    let mut c = x2.poly_p::<P, _>(&[
        4.16666666666665929218E-2,
        -1.38888888888730564116E-3,
        2.48015872888517045348E-5,
        -2.75573141792967388112E-7,
        2.08757008419747316778E-9,
        -1.13585365213876817300E-11,
    ]);

    s = s.mul_adde(x2 * x, x); // s = x + (x * x2) * s;
    c = c.mul_adde(x4, x2.nmul_adde(V::HALF, V::ONE)); // c = 1.0 - x2 * 0.5 + (x2 * x2) * c;

    // swap sin and cos if odd quadrant
    let swap = (q & V::Bits::ONE).cmp_ne(V::Bits::ZERO);

    if P::POLICY.check_overflow {
        let overflow = y.cmp_gt(V::splat((1u64 << 52) as f64 - 1.0)) & xa.is_finite();

        s = overflow.select(V::ZERO, s);
        c = overflow.select(V::ONE, c);
    }

    let sin1 = swap.select(c, s);
    let cos1 = swap.select(s, c);

    let signsin = V::from_bits(q << 62) ^ xx;
    let signcos = V::from_bits(((q + V::Bits::ONE) & V::Bits::splat(2)) << 62);

    // combine signs
    (sin1.mul_sign(signsin), cos1 ^ signcos)
}
