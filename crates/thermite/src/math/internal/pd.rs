use crate::{
    divider::Divider,
    math::{consts::FloatConsts as _, policy::policies::ExtraPrecision},
};
use core::f64::consts::{FRAC_1_PI, LN_10, LOG2_E, SQRT_2};

use super::*;

impl<R> MathInternal<f64> for R
where
    R: FloatRegister<Element = f64>,
{
    #[inline(always)]
    fn sincos<P: Policy>(xx: Vf<Self>) -> (Vf<Self>, Vf<Self>) {
        let dp1 = Vf::splat(7.853981554508209228515625E-1 * 2.0);
        let dp2 = Vf::splat(7.94662735614792836714E-9 * 2.0);
        let dp3 = Vf::splat(3.06161699786838294307E-17 * 2.0);
        let xa = xx.abs();

        let y = (xa * Vf::FRAC_2_PI).round();
        let q = Vu::<R>::fast_from(y);
        //let q = unsafe { y.to_uint_fast() };

        // Reduce by extended precision modular arithmetic
        // x = ((xa - y * DP1F) - y * DP2F) - y * DP3F;
        let x = y.nmul_add(dp3, y.nmul_add(dp2, y.nmul_add(dp1, xa)));

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
        c = c.mul_adde(x4, x2.nmul_adde(Vf::HALF, Vf::ONE)); // c = 1.0 - x2 * 0.5 + (x2 * x2) * c;

        // swap sin and cos if odd quadrant
        let swap = (q & Vu::<R>::ONE).cmp_ne(Vu::<R>::ZERO);

        if P::POLICY.check_overflow {
            let overflow = y.cmp_gt(Vf::splat((1u64 << 52) as f64 - 1.0)) & xa.is_finite();

            let s = overflow.select(Vf::ZERO, s);
            let c = overflow.select(Vf::ONE, c);
        }

        let sin1 = swap.select(c, s);
        let cos1 = swap.select(s, c);

        let signsin = Vf::from_bits(q << 62) ^ xx;
        let signcos = Vf::from_bits(((q + Vu::<R>::ONE) & Vu::<R>::splat(2)) << 62);

        // combine signs
        (sin1.mul_sign(signsin), cos1 ^ signcos)
    }

    #[inline(always)]
    fn sinh<P: Policy>(x0: Vf<Self>) -> Vf<Self> {
        let x = x0.abs();

        let x_small = x.cmp_le(Vf::ONE);

        let mut y2 = Vf::EMPTY;

        if P::POLICY.avoid_branching || !x_small.all() {
            y2 = x.exph_p::<P>();
            y2 -= Vf::splat(0.25) / y2;

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
    fn cosh<P: Policy>(x0: Vf<Self>) -> Vf<Self> {
        let y = Self::exph::<P>(x0.abs());
        y + Vf::splat(0.25) / y
    }

    #[inline(always)]
    fn tanh<P: Policy>(x0: Vf<Self>) -> Vf<Self> {
        let x = x0.abs();

        let x_small = x.cmp_le(Vf::splat(0.625));

        let mut y2 = Vf::EMPTY;

        if P::POLICY.avoid_branching || !x_small.all() {
            y2 = (x + x).exp_p::<P>();
            y2 = (y2 - Vf::ONE) / (y2 + Vf::ONE); // originally (1 - 2/(y2 + 1))

            if P::POLICY.check_overflow {
                y2 = x.cmp_gt(Vf::splat(350.0)).select(Vf::ONE, y2);
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
    fn asin<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        asin_internal::<Self, P, false>(x)
    }

    #[inline(always)]
    fn acos<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        asin_internal::<Self, P, true>(x)
    }

    #[inline(always)]
    fn atan<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        atan_internal::<Self, P, false>(x, Vf::ZERO)
    }

    #[inline(always)]
    fn atan2<P: Policy>(y: Vf<Self>, x: Vf<Self>) -> Vf<Self> {
        atan_internal::<Self, P, true>(y, x)
    }

    #[inline(always)]
    fn asinh<P: Policy>(x0: Vf<Self>) -> Vf<Self> {
        let x = x0.abs();
        let x2 = x * x;

        let x_small = x.cmp_le(Vf::splat(0.533));

        let mut y2 = Vf::EMPTY;

        if P::POLICY.avoid_branching || !x_small.all() {
            y2 = ((x2 + Vf::ONE).sqrt() + x).ln_p::<P>();

            if const { P::POLICY.check_overflow || !P::POLICY.avoid_precision_branches() } {
                let x_huge = x.cmp_gt(Vf::splat(1e20));

                if crate::unlikely(x_huge.any()) {
                    y2 = x_huge.select(x.ln_p::<P>() + Vf::LN_2, y2);
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
    fn acosh<P: Policy>(x0: Vf<Self>) -> Vf<Self> {
        let x1 = x0 - Vf::ONE;

        let x_small = x1.cmp_le(Vf::splat(0.49));

        let mut y2 = Vf::EMPTY;

        if P::POLICY.avoid_branching || !x_small.all() {
            y2 = (x0.mul_sube(x0, Vf::ONE).sqrt() + x0).ln_p::<P>();

            if const { P::POLICY.check_overflow && !P::POLICY.avoid_precision_branches() } {
                let x_huge = x1.cmp_gt(Vf::splat(1e20));

                if crate::unlikely(x_huge.any()) {
                    y2 = x_huge.select(x0.ln_p::<P>() + Vf::LN_2, y2);
                }
            }

            if const { P::POLICY.avoid_precision_branches() } {
                // certain overflow checks can still be important even if precision is not
                if P::POLICY.check_overflow {
                    y2 = x0.cmp_lt(Vf::ONE).select(Vf::NAN, y2);
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
                y1 = x0.cmp_lt(Vf::ONE).select(Vf::NAN, y1);
            }

            y2 = x_small.select(y1, y2);
        }

        y2
    }

    #[inline(always)]
    fn atanh<P: Policy>(x0: Vf<Self>) -> Vf<Self> {
        let x = x0.abs();

        let x_small = x.cmp_le(Vf::HALF);

        let mut y2 = Vf::EMPTY;

        if P::POLICY.avoid_branching || !x_small.all() {
            y2 = ((Vf::ONE + x) / (Vf::ONE - x)).ln_p::<P>() * Vf::HALF;

            if P::POLICY.check_overflow {
                let y3 = x.cmp_eq(Vf::ONE).select(Vf::INFINITY, Vf::NAN);
                y2 = x.cmp_ge(Vf::ONE).select(y3, y2);
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
    fn exp<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        exp_d_internal::<Self, P, EXP_MODE_EXP>(x)
    }

    #[inline(always)]
    fn exph<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        exp_d_internal::<Self, P, EXP_MODE_EXPH>(x)
    }

    #[inline(always)]
    fn exp2<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        exp_d_internal::<Self, P, EXP_MODE_POW2>(x)
    }

    #[inline(always)]
    fn exp10<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        exp_d_internal::<Self, P, EXP_MODE_POW10>(x)
    }

    #[inline(always)]
    fn exp_m1<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        exp_d_internal::<Self, P, EXP_MODE_EXPM1>(x)
    }

    #[inline(always)]
    fn powf<P: Policy>(x0: Vf<Self>, y: Vf<Self>) -> Vf<Self> {
        // define constants
        let ln2d_hi = Vf::splat(0.693145751953125); // log(2) in extra precision, high bits
        let ln2d_lo = Vf::splat(1.42860682030941723212E-6); // low bits of log(2)

        let x1 = x0.abs();

        let mut x = fraction2(x1);

        let blend = x.cmp_gt(Vf::splat(SQRT_2 / 2.0));

        x += blend.andnot(x);
        x -= Vf::ONE;

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

        let ef = exponent_f(x1) + (blend.value() & Vf::ONE);

        // multiply exponent by y, nearest integer e1 goes into exponent of result, remainder yr is added to log
        let e1 = (ef * y).round();
        let yr = ef.mul_sube(y, e1); // calculate remainder yr. precision very important here

        // add initial terms to expansion
        let lg = Vf::HALF.nmul_adde(x2, x) + lg1; // lg = (x - 0.5f * x2) + lg1;

        // calculate rounding errors in lg
        // rounding error in multiplication 0.5*x*x
        let x2err = (Vf::HALF * x).mul_sube(x, Vf::HALF * x2);

        // rounding error in additions and subtractions
        let lgerr = Vf::HALF.mul_adde(x2, lg - x) - lg1; // lgerr = ((lg - x) + 0.5f * x2) - lg1;

        // extract something for the exponent
        let e2 = (lg * y * Vf::LOG2_E).round();

        // subtract this from lg, with extra precision
        let mut v = e2.nmul_adde(ln2d_lo, lg.mul_sube(y, e2 * ln2d_hi));

        // add remainder from ef * y
        v = yr.mul_adde(Vf::LN_2, v); // v += yr * VM_LN2;

        // correct for previous rounding errors
        v = (lgerr + x2err).nmul_adde(y, v); // v -= (lgerr + x2err) * y;

        // extract something for the exponent if possible
        let mut x = v;
        let e3 = (x * Vf::LOG2_E).round();

        // high precision multiplication not needed here because abs(e3) <= 1
        x = e3.nmul_adde(Vf::LN_2, x); // x -= e3 * VM_LN2;

        // Taylor coefficients for exp function, 1/n!
        let mut z = x.poly_p::<P, _>(&[
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
        let ei: Vs<R> = ee.fast_cast();

        // biased exponent of result:
        let ej = ei + (Vs::<R>::from_bits(x.abs()) >> 52);

        // add exponent by signed integer addition
        let mut z = Vf::<R>::from_bits(Vs::<R>::from_bits(z) + (ei << 52));

        if !P::POLICY.check_overflow {
            return z;
        }

        // check exponent for overflow and underflow
        let overflow = ej.cmp_ge(Vs::<R>::splat(0x07FF)).cast() | ee.cmp_gt(Vf::splat(3000.0));
        let underflow = ej.cmp_le(Vs::<R>::splat(0x0000)).cast() | ee.cmp_lt(Vf::splat(-3000.0));

        // check for special cases
        let xfinite = x0.is_finite();
        let yfinite = y.is_finite();
        let efinite = ee.is_finite();

        let xzero = x0.is_zero_or_subnormal();
        let xsign = x0.is_negative();

        if crate::unlikely((overflow | underflow).any()) {
            z = underflow.select(Vf::ZERO, z);
            z = overflow.select(Vf::INFINITY, z);
        }

        let yzero = y.cmp_eq(Vf::ZERO);
        let yneg = y.cmp_lt(Vf::ZERO);

        // pow_case_x0
        z = xzero.select(yneg.select(Vf::INFINITY, yzero.select(Vf::ONE, Vf::ZERO)), z);

        let mut yodd = Vf::ZERO;

        if xsign.any() {
            let yint = y.cmp_eq(y.round());
            yodd = y << 63;

            let z1 = yint.select(z | yodd, x0.cmp_eq(Vf::ZERO).select(z, Vf::NAN));

            yodd = yint.select(yodd, Vf::ZERO);

            z = xsign.select(z1, z);
        }

        let not_special = (xfinite & yfinite & (efinite | xzero));

        if crate::likely(not_special.all()) {
            return z; // fast return
        }

        // handle special error cases: y infinite
        let z1 = (yfinite & efinite).select(
            z,
            x1.cmp_eq(Vf::ONE).select(
                Vf::ONE,
                (x1.cmp_gt(Vf::ONE) ^ y.is_negative()).select(Vf::INFINITY, Vf::ZERO),
            ),
        );

        // handle x infinite
        let z1 = xfinite.select(
            z1,
            yzero.select(
                Vf::ONE,
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
    fn cbrt<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        let b1 = Vu::<Self>::splat(715094163); // B1 = (1023-1023/3-0.03306235651)*2**20
        let b2 = Vu::<Self>::splat(696219795); // B2 = (1023-1023/3-54/3-0.03306235651)*2**20
        let m = Vu::<Self>::splat(0x7fffffff); // u32::MAX >> 1

        let x1p54 = x * Vf::<Self>::splat(f64::from_bits(0x4350000000000000)); // 0x1p54 === 2 ^ 54

        let hx0 = (x.into_bits() >> 32) & m;

        let x_small = hx0.cmp_lt(Vu::<Self>::splat(0x00100000));

        let xs = x_small.select(x1p54, x); // note that this upcasts
        let b = x_small.select(b2, b1);

        let mut ui = xs.into_bits();
        let mut hx = (ui >> 32) & m;

        // NOTE: Using the branched divider with a constant
        // leads to better codegen when the branch is inlined.
        hx = hx / Divider::u64(3) + b;

        ui &= Vu::<Self>::splat(1 << 63);
        ui |= hx << 32;

        let mut t = Vf::<Self>::from_bits(ui);

        let r = (t * t) * (t / x); // encourage ILP
        let r2 = r * r;

        t *= r.poly_p::<P, _>(&[
            1.87595182427177009643,   /* 0x3ffe03e6, 0x0f61e692 */
            -1.88497979543377169875,  /* 0xbffe28e0, 0x92f02420 */
            1.621429720105354466140,  /* 0x3ff9f160, 0x4a49d6c2 */
            -0.758397934778766047437, /* 0xbfe844cb, 0xbee751d9 */
            0.145996192886612446982,  /* 0x3fc2b000, 0xd4e4edd7 */
        ]);

        ui = t.into_bits();
        ui = (ui + Vu::<Self>::splat(0x80000000)) & Vu::<Self>::splat(0xffffffffc0000000);
        t = Vf::<Self>::from_bits(ui);

        let r = if const { P::POLICY.precision.ge(PrecisionPolicy::Best) || !Self::HAS_TRUE_FMA } {
            // original form, 5 simple ops, 2 divisions
            ((x / (t * t)) - t) / ((t + t) + (x / (t * t)))
        } else {
            // fast form, 3 simple ops, 1 division, 1 fma
            let t3 = t * t * t;
            (x - t3) / t3.mul_add(Vf::<Self>::TWO, x)
        };

        t = r.mul_adde(t, t);

        if !P::POLICY.check_overflow {
            return x.cmp_eq(Vf::<Self>::ZERO).select(x, t);
        }

        (hx0.cmp_gt(Vu::<Self>::splat(0x7f800000)) | hx0.cmp_eq(Vu::<Self>::ZERO)).select(x, t)
    }

    #[inline(always)]
    fn ln<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        ln_d_internal::<Self, P, false>(x)
    }

    #[inline(always)]
    fn ln1p<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        ln_d_internal::<Self, P, true>(x)
    }

    #[inline(always)]
    fn log2<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        ln_d_internal::<Self, P, false>(x) * Vf::LOG2_E
    }

    #[inline(always)]
    fn log10<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        ln_d_internal::<Self, P, false>(x) * Vf::LOG10_2
    }

    #[inline(always)]
    fn ln1m_expnx<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        (Vf::ONE - (-x).exp_p::<P>()).ln_p::<P>()
    }

    #[inline(always)]
    fn ln1m_expnx_ext<P: Policy>(x: Vf<Self>, _lnx: Vf<Self>) -> Vf<Self> {
        (Vf::ONE - (-x).exp_p::<P>()).ln_p::<P>()
    }

    #[inline(always)]
    fn erf<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        let x2 = x * x;
        let res = x * x2.poly_rational_p::<P, _, _>(
            &[
                5.55923013010394962768e4,
                7.00332514112805075473e3,
                2.23200534594684319226e3,
                9.00260197203842689217e1,
                9.60497373987051638749e0,
                0.0,
            ],
            &[
                4.92673942608635921086e4,
                2.26290000613890934246e4,
                4.59432382970980127987e3,
                5.21357949780152679795e2,
                3.35617141647503099647e1,
                1.00000000000000000000e0,
            ],
        );

        if P::POLICY.check_overflow {
            // x^2 highest point in the polynomial, use x2 to avoid needing absolute value
            // TODO: Find more exact value?
            x2.cmp_gt(Vf::splat(8.135455562428929)).select(x.signum(), res)
        } else {
            res
        }
    }

    #[inline(always)]
    fn erfinv<P: Policy>(y: Vf<Self>) -> Vf<Self> {
        let a = y.abs();

        let w = -a.nmul_adde(a, Vf::ONE).ln_p::<P>();

        // https://www.desmos.com/calculator/yduhxx1ukm values extracted via JS console
        let mut p0 = (w - Vf::splat(2.5)).poly_p::<P, _>(&[
            1.501409350414994,
            0.2466402709383954,
            -0.0041773392840529855,
            -0.001252754693878528,
            0.00021818504236422313,
            -0.000005055953518603739,
            -0.000003451228003698613,
            4.691555466910589e-7,
            1.565009183876413e-8,
            -7.498144332533493e-9,
            2.378447620687541e-9,
            4.340759057762667e-10,
            -1.1526825105953649e-11,
            -3.605158594283844e-12,
        ]);

        let w_big = w.cmp_ge(Vf::splat(5.0)); // at around |x| > 0.99662533231, so unlikely

        if P::POLICY.avoid_branching || crate::unlikely(w_big.any()) {
            let mut p1 = (w.sqrt() - Vf::splat(3.0)).poly_p::<P, _>(&[
                2.914513093490991,
                1.5466942804733321,
                1.5950004257395263,
                2.559965578101086,
                2.3489887347568135,
                0.7600225853251197,
                -0.9258061028319879,
                -1.574375166164548,
                -1.2294848322739875,
                -0.6192716293714041,
                -0.21681459128064842,
                -0.05369968979686224,
                -0.009288117987439485,
                -0.0010722580888930223,
                -0.00007449590390143766,
                -0.0000023620166848468398,
            ]);

            if P::POLICY.check_overflow {
                p1 = a.cmp_eq(Vf::ONE).select(Vf::INFINITY, p1); // erfinv(x == 1) = inf
                p1 = a.cmp_gt(Vf::ONE).select(Vf::NAN, p1); // erfinv(x > 1) = NaN
            }

            p0 = w_big.select(p1, p0);
        }

        p0 * y
    }
}

#[inline(always)]
fn fraction2<R: MathInternal<f64>>(x: Vf<R>) -> Vf<R> {
    // set exponent to 0 + bias
    (x & Vf::splat(f64::from_bits(0x000FFFFFFFFFFFFF))) | Vf::splat(f64::from_bits(0x3FE0000000000000))
}

#[inline(always)]
fn exponent<R: MathInternal<f64>>(x: Vf<R>) -> Vs<R> {
    // shift out sign, extract exp, subtract bias
    Vs::<R>::from_bits((Vu::<R>::from_bits(x) << 1) >> 53) - Vs::<R>::splat(0x3FF)
}

#[inline(always)]
fn exponent_f<R: MathInternal<f64>>(x: Vf<R>) -> Vf<R> {
    let pow2_52 = Vf::<R>::splat(4503599627370496.0);
    let bias = Vf::<R>::splat(1023.0);

    Vf::from_bits((Vu::<R>::from_bits(x) >> 52) | pow2_52.into_bits()) - (pow2_52 + bias)
}

#[inline(always)]
fn ln_d_internal<R: MathInternal<f64>, P: Policy, const P1: bool>(x0: Vf<R>) -> Vf<R> {
    let ln2_hi = Vf::splat(0.693359375);
    let ln2_lo = Vf::splat(-2.121944400546905827679E-4);
    let x1 = if P1 { x0 + Vf::ONE } else { x0 };

    let mut x = fraction2::<R>(x1);
    let mut fe = Vf::from(exponent::<R>(x1));

    let blend = x.cmp_gt(Vf::splat(SQRT_2 * 0.5));

    x = blend.select(x, x + x); // x = x.conditional_add(x, !blend);
    fe = blend.select(fe + Vf::ONE, fe); // fe = fe.conditional_add(Vf::ONE, blend);

    let xp1 = x - Vf::ONE;

    x = if P1 {
        // log(x+1). Avoid loss of precision when adding 1 and later subtracting 1 if exponent = 0
        fe.cmp_eq(Vf::ZERO).select(x0, xp1)
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
    res += x2.nmul_adde(Vf::HALF, x); // res += x - 0.5 * x2;
    res = fe.mul_adde(ln2_hi, res); // res += fe * ln2_hi;

    if !P::POLICY.check_overflow {
        return res;
    }

    let overflow = !x1.is_finite();
    let underflow = x1.cmp_lt(Vf::splat(2.2250738585072014E-308));

    if !P::POLICY.avoid_branching && crate::likely((overflow | underflow).none()) {
        return res;
    }

    res = underflow.select(Vf::NAN, res); // x1 < 0 gives NAN
    res = x1.is_zero_or_subnormal().select(Vf::NEG_INFINITY, res); // x1 == 0 gives -INF
    res = overflow.select(x1, res); // INF or NAN goes through
    res = (x1.is_infinite() & x1.is_negative()).select(Vf::NAN, res); // -INF gives NAN

    res
}

#[inline(always)]
fn atan_internal<R: MathInternal<f64>, P: Policy, const ATAN2: bool>(y: Vf<R>, x: Vf<R>) -> Vf<R> {
    let morebits = Vf::splat(6.123233995736765886130E-17);
    let morebitso2 = Vf::splat(6.123233995736765886130E-17 * 0.5);
    let t3po8 = Vf::splat(SQRT_2 + 1.0);

    let mut swapxy = Mask::FALSY;

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
                x2 = both_inf.select(x2 & Vf::NEG_ONE, x2);
                y2 = both_inf.select(y2 & Vf::NEG_ONE, y2);
            }
        }

        y2 / x2
    } else {
        y.abs()
    };

    let not_big = t.cmp_le(t3po8);
    let not_small = t.cmp_ge(Vf::splat(0.66));

    let s = not_big.select(Vf::FRAC_PI_4, Vf::FRAC_PI_2) & not_small.value();

    let fac = not_big.select(morebitso2, morebits) & not_small.value();

    let a = (not_big.value() & t) + (not_small.value() & Vf::NEG_ONE);
    let b = (not_big.value() & Vf::ONE) + (not_small.value() & t);

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
    let mut re = re0.mul_adde(z * zz, z + s + fac);

    if ATAN2 {
        re = swapxy.select(Vf::FRAC_PI_2 - re, re);
        re = (x | y).cmp_eq(Vf::ZERO).select(Vf::ZERO, re); // atan2(0,0) = 0 by convention
        // also for x = -0.
        re = x.select_negative(Vf::PI - re, re);
    }

    re.mul_sign(y)
}

#[inline(always)]
fn asin_internal<R: MathInternal<f64>, P: Policy, const ACOS: bool>(x: Vf<R>) -> Vf<R> {
    let xa = x.abs();

    let is_big = xa.cmp_ge(Vf::splat(0.625));

    let x1 = is_big.select(Vf::ONE - xa, xa * xa);

    let x2 = x1 * x1;
    let x4 = x2 * x2;
    let x8 = x4 * x4;

    let undef = Vf::EMPTY;

    let mut px = undef;
    let mut qx = undef;
    let mut rx = undef;
    let mut sx = undef;
    let mut xb = undef;

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
        let z1 = x.select_negative(Vf::PI - z1, z1);
        let z2 = Vf::FRAC_PI_2 - z2.mul_sign(x);
        is_big.select(z1, z2)
    } else {
        let z1 = Vf::FRAC_PI_2 - z1;
        is_big.select(z1, z2).mul_sign(x)
    }
}

#[inline(always)]
fn pow2n_d<R: MathInternal<f64>>(n: Vf<R>) -> Vf<R> {
    let pow2_52 = Vf::splat(4503599627370496.0);
    let bias = Vf::splat(1023.0);

    (n + (bias + pow2_52)) << 52
}

#[inline(always)]
fn exp_d_internal<R: MathInternal<f64>, P: Policy, const MODE: u8>(x0: Vf<R>) -> Vf<R> {
    let mut x = x0;
    let mut r;

    let max_x;

    match MODE {
        EXP_MODE_POW2 => {
            max_x = 1022.0;

            r = x0.round();

            x -= r;
            x *= Vf::LN_2;
        }
        EXP_MODE_POW10 => {
            max_x = 307.65;

            let log10_2_hi = Vf::splat(0.30102999554947019); // log10(2) in two parts
            let log10_2_lo = Vf::splat(1.1451100899212592E-10);

            r = (x0 * Vf::splat(LN_10 * LOG2_E)).round();

            x = r.nmul_adde(log10_2_hi, x); // x -= r * log10_2_hi;
            x = r.nmul_adde(log10_2_lo, x); // x -= r * log10_2_lo;
            x *= Vf::LN_10;
        }
        _ => {
            max_x = const { if MODE == EXP_MODE_EXP { 708.39 } else { 709.7 } };

            let ln2d_hi = Vf::splat(0.693145751953125);
            let ln2d_lo = Vf::splat(1.42860682030941723212E-6);

            r = (x0 * Vf::splat(LOG2_E)).round();

            x = r.nmul_adde(ln2d_hi, x); // x -= r * ln2_hi;
            x = r.nmul_adde(ln2d_lo, x); // x -= r * ln2_lo;

            if MODE == EXP_MODE_EXPH {
                r -= Vf::ONE;
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

    let n2 = pow2n_d::<R>(r);

    z = match MODE {
        EXP_MODE_EXPM1 => z.mul_adde(n2, n2 - Vf::ONE),
        _ => z.mul_adde(n2, n2), // (z + 1.0f) * n2
    };

    if P::POLICY.check_overflow {
        let in_range = x0.abs().cmp_lt(Vf::splat(max_x)) & x0.is_finite();

        if crate::likely(in_range.all()) {
            return z;
        }

        let underflow_value = const { if MODE == EXP_MODE_EXPM1 { Vf::NEG_ONE } else { Vf::ZERO } };

        r = x0.select_negative(underflow_value, Vf::INFINITY);
        z = in_range.select(z, r);
        z = x0.is_nan().select(x0, z);
    }

    z
}
