use super::{poly::*, *};

use core::f64::consts::{LN_10, LN_2, LOG2_E, SQRT_2};

impl<S: Simd> SimdVectorizedMathInternal<S> for f64
where
    <S as Simd>::Vf64: SimdFloatVector<S, Element = f64>,
{
    const __EPSILON: Self = f64::EPSILON;
    const __SQRT_EPSILON: Self = 1.4901161193847656314265919999999999861416556075118966152884e-8;
    const __DIGITS: u32 = f64::MANTISSA_DIGITS;

    #[inline(always)]
    fn sin_cos<P: Policy>(xx: Self::Vf) -> (Self::Vf, Self::Vf) {
        /*
        // This ended up being pointless, but I'm keeping the code around for reference.

        if P::POLICY.precision == PrecisionPolicy::Reference {
            let epsilon = Vf64::<S>::splat(f64::EPSILON * 0.5);

            let xx2 = xx * xx;
            let mut n = 1;
            let mut fact = 1.0;
            let mut powers = xx2; // NOTE: powers is always positive

            let mut sin_sum = xx;
            let mut cos_sum = Vf64::<S>::one();

            for _ in 0..P::POLICY.max_series_iterations {
                let sign = if n & 1 == 0 {
                    Vf64::<S>::zero()
                } else {
                    Vf64::<S>::neg_zero()
                };

                let n2 = n + n;
                fact *= n2 as f64 * (n2 - 1) as f64;

                let cos_delta = powers / Vf64::<S>::splat(fact);
                let sin_delta = (powers * xx) / Vf64::<S>::splat(fact * (n2 + 1) as f64);
                cos_sum += sign ^ cos_delta;
                sin_sum += sign ^ sin_delta;

                // the cosine term is always positive, but sine is odd and can be negative
                let not_converged = cos_delta.gt(epsilon) | sin_delta.abs().gt(epsilon);

                if not_converged.none() {
                    return (sin_sum, cos_sum);
                }

                powers *= xx2;
                n += 1;
            }

            // panic!("sin_cos did not converge");
        }
        */

        let dp1 = Vf64::<S>::splat(7.853981554508209228515625E-1 * 2.0);
        let dp2 = Vf64::<S>::splat(7.94662735614792836714E-9 * 2.0);
        let dp3 = Vf64::<S>::splat(3.06161699786838294307E-17 * 2.0);
        let zero = Vf64::<S>::zero();
        let one = Vf64::<S>::one();

        let xa = xx.abs();

        let y = (xa * Vf64::<S>::FRAC_2_PI()).round();
        let q = unsafe { y.to_uint_fast() };

        // Reduce by extended precision modular arithmetic
        // x = ((xa - y * DP1F) - y * DP2F) - y * DP3F;
        let x = y.nmul_add(dp3, y.nmul_add(dp2, y.nmul_add(dp1, xa)));

        // Taylor expansion of sin and cos, valid for -pi/4 <= x <= pi/4
        let x2 = x * x;
        let x4 = x2 * x2;

        let mut s = x2.poly_p::<P>(&[
            -1.66666666666666307295E-1,
            8.33333333332211858878E-3,
            -1.98412698295895385996E-4,
            2.75573136213857245213E-6,
            -2.50507477628578072866E-8,
            1.58962301576546568060E-10,
        ]);

        let mut c = x2.poly_p::<P>(&[
            4.16666666666665929218E-2,
            -1.38888888888730564116E-3,
            2.48015872888517045348E-5,
            -2.75573141792967388112E-7,
            2.08757008419747316778E-9,
            -1.13585365213876817300E-11,
        ]);

        s = s.mul_adde(x2 * x, x); // s = x + (x * x2) * s;
        c = c.mul_adde(x4, x2.nmul_adde(Vf64::<S>::splat(0.5), one)); // c = 1.0 - x2 * 0.5 + (x2 * x2) * c;

        // swap sin and cos if odd quadrant
        let swap = (q & Vu64::<S>::one()).ne(Vu64::<S>::zero());

        if P::POLICY.check_overflow {
            let overflow = y.gt(Vf64::<S>::splat((1u64 << 52) as f64 - 1.0)) & xa.is_finite();

            let s = overflow.select(zero, s);
            let c = overflow.select(one, c);
        }

        let sin1 = swap.select(c, s);
        let cos1 = swap.select(s, c);

        let signsin = Vf64::<S>::from_bits((q << 62)) ^ xx;
        let signcos = Vf64::<S>::from_bits(((q + Vu64::<S>::one()) & Vu64::<S>::splat(2)) << 62);

        // combine signs
        (sin1.combine_sign(signsin), cos1 ^ signcos)
    }

    #[inline(always)]
    fn asin<P: Policy>(x: Self::Vf) -> Self::Vf {
        asin_internal::<S, P, false>(x)
    }

    #[inline(always)]
    fn acos<P: Policy>(x: Self::Vf) -> Self::Vf {
        asin_internal::<S, P, true>(x)
    }

    #[inline(always)]
    fn atan<P: Policy>(x: Self::Vf) -> Self::Vf {
        atan_internal::<S, P, false>(x, unsafe { Vf64::<S>::undefined() })
    }
    #[inline(always)]
    fn atan2<P: Policy>(y: Self::Vf, x: Self::Vf) -> Self::Vf {
        atan_internal::<S, P, true>(y, x)
    }

    #[inline(always)]
    fn sinh<P: Policy>(x0: Self::Vf) -> Self::Vf {
        let one = Vf64::<S>::one();

        let x = x0.abs();

        let x_small = x.le(one);

        let mut y1 = unsafe { Vf64::<S>::undefined() };
        let mut y2 = unsafe { Vf64::<S>::undefined() };

        // use bitmask directly to avoid two calls
        let bitmask = x_small.bitmask();

        // if not all are small
        if P::POLICY.avoid_branching || !bitmask.all() {
            y2 = x.exph_p::<P>();
            y2 -= Vf64::<S>::splat(0.25) / y2;

            if P::POLICY.avoid_precision_branches() {
                return y2.combine_sign(x0);
            }
        }

        // if any are small
        if P::POLICY.avoid_branching || bitmask.any() {
            let x2 = x * x;

            y1 = x2.poly_p::<P>(&[
                -3.51754964808151394800E5,
                -1.15614435765005216044E4,
                -1.63725857525983828727E2,
                -7.89474443963537015605E-1,
            ]) / x2.poly_p::<P>(&[
                -2.11052978884890840399E6,
                3.61578279834431989373E4,
                -2.77711081420602794433E2,
                1.0,
            ]);

            y1 = y1.mul_adde(x * x2, x);
        }

        x_small.select(y1, y2).combine_sign(x0)
    }

    #[inline(always)]
    fn tanh<P: Policy>(x0: Self::Vf) -> Self::Vf {
        let one = Vf64::<S>::one();

        let x = x0.abs();

        let x_small = x.le(Vf64::<S>::splat(0.625));

        let mut y1 = unsafe { Vf64::<S>::undefined() };
        let mut y2 = unsafe { Vf64::<S>::undefined() };

        // use bitmask directly to avoid two calls
        let bitmask = x_small.bitmask();

        // if not all are small
        if P::POLICY.avoid_branching || !bitmask.all() {
            y2 = (x + x).exp_p::<P>();
            y2 = (y2 - one) / (y2 + one); // originally (1 - 2/(y2 + 1)), but doing it this way avoids loading 2.0

            if P::POLICY.check_overflow {
                y2 = x.gt(Vf64::<S>::splat(350.0)).select(one, y2);
            }

            if P::POLICY.avoid_precision_branches() {
                return y2.combine_sign(x0);
            }
        }

        // if any are small
        if P::POLICY.avoid_branching || bitmask.any() {
            let x2 = x * x;

            y1 = x2.poly_p::<P>(&[
                -1.61468768441708447952E3,
                -9.92877231001918586564E1,
                -9.64399179425052238628E-1,
            ]) / x2.poly_p::<P>(&[
                4.84406305325125486048E3,
                2.23548839060100448583E3,
                1.12811678491632931402E2,
                1.0,
            ]);

            y1 = y1.mul_adde(x2 * x, x);
        }

        x_small.select(y1, y2).combine_sign(x0)
    }

    #[inline(always)]
    fn asinh<P: Policy>(x0: Self::Vf) -> Self::Vf {
        let one = Vf64::<S>::one();

        let x = x0.abs();
        let x2 = x0 * x0;

        let x_small = x.le(Vf64::<S>::splat(0.533));

        let mut y1 = unsafe { Vf64::<S>::undefined() };
        let mut y2 = unsafe { Vf64::<S>::undefined() };

        let bitmask = x_small.bitmask();

        if P::POLICY.avoid_branching || !bitmask.all() {
            y2 = ((x2 + one).sqrt() + x).ln_p::<P>();

            if P::POLICY.check_overflow {
                let x_huge = x.gt(Vf64::<S>::splat(1e20));

                if thermite_unlikely!(x_huge.any()) {
                    y2 = x_huge.select(x.ln_p::<P>() + Vf64::<S>::LN_2(), y2);
                }
            }

            if P::POLICY.avoid_precision_branches() {
                return y2.combine_sign(x0);
            }
        }

        if P::POLICY.avoid_branching || bitmask.any() {
            y1 = x2.poly_p::<P>(&[
                -5.56682227230859640450E0,
                -9.09030533308377316566E0,
                -4.37390226194356683570E0,
                -5.91750212056387121207E-1,
                -4.33231683752342103572E-3,
            ]) / x2.poly_p::<P>(&[
                3.34009336338516356383E1,
                6.95722521337257608734E1,
                4.86042483805291788324E1,
                1.28757002067426453537E1,
                1.0,
            ]);

            y1 = y1.mul_adde(x2 * x, x);
        }

        x_small.select(y1, y2).combine_sign(x0)
    }

    #[inline(always)]
    fn acosh<P: Policy>(x0: Self::Vf) -> Self::Vf {
        let one = Vf64::<S>::one();

        let x1 = x0 - one;

        let x_small = x1.lt(Vf64::<S>::splat(0.49));

        let mut y1 = unsafe { Vf64::<S>::undefined() };
        let mut y2 = unsafe { Vf64::<S>::undefined() };

        let bitmask = x_small.bitmask();

        if P::POLICY.avoid_branching || !bitmask.all() {
            y2 = (x0.mul_sube(x0, one).sqrt() + x0).ln_p::<P>();

            if P::POLICY.check_overflow {
                let x_huge = x1.gt(Vf64::<S>::splat(1e20));
                if thermite_unlikely!(x_huge.any()) {
                    y2 = x_huge.select(x0.ln_p::<P>() + Vf64::<S>::LN_2(), y2);
                }
            }

            if P::POLICY.avoid_precision_branches() {
                return y2;
            }
        }

        if P::POLICY.avoid_branching || bitmask.any() {
            y1 = x1.sqrt()
                * x1.poly_p::<P>(&[
                    1.10855947270161294369E5,
                    1.08102874834699867335E5,
                    3.43989375926195455866E4,
                    3.94726656571334401102E3,
                    1.18801130533544501356E2,
                ])
                / x1.poly_p::<P>(&[
                    7.83869920495893927727E4,
                    8.29725251988426222434E4,
                    2.97683430363289370382E4,
                    4.15352677227719831579E3,
                    1.86145380837903397292E2,
                    1.0,
                ]);

            if P::POLICY.check_overflow {
                y1 = x0.lt(one).select(Vf64::<S>::nan(), y1);
            }
        }

        x_small.select(y1, y2)
    }

    #[inline(always)]
    fn atanh<P: Policy>(x0: Self::Vf) -> Self::Vf {
        let one = Vf64::<S>::one();
        let half = Vf64::<S>::splat(0.5);

        let x = x0.abs();

        let x_small = x.lt(half);

        let mut y1 = unsafe { Vf64::<S>::undefined() };
        let mut y2 = unsafe { Vf64::<S>::undefined() };

        let bitmask = x_small.bitmask();

        if P::POLICY.avoid_branching || !bitmask.all() {
            y2 = ((one + x) / (one - x)).ln_p::<P>() * half;

            if P::POLICY.check_overflow {
                y2 = x
                    .gt(one)
                    .select(x.eq(one).select(Vf64::<S>::infinity(), Vf64::<S>::nan()), y2);
            }

            if P::POLICY.avoid_precision_branches() {
                return y2.combine_sign(x0);
            }
        }

        if P::POLICY.avoid_branching || bitmask.any() {
            let x2 = x * x;

            y1 = x2.poly_p::<P>(&[
                -3.09092539379866942570E1,
                6.54566728676544377376E1,
                -4.61252884198732692637E1,
                1.20426861384072379242E1,
                -8.54074331929669305196E-1,
            ]) / x2.poly_p::<P>(&[
                -9.27277618139601130017E1,
                2.52006675691344555838E2,
                -2.49839401325893582852E2,
                1.08938092147140262656E2,
                -1.95638849376911654834E1,
                1.0,
            ]);

            y1 = y1.mul_adde(x2 * x, x);
        }

        x_small.select(y1, y2).combine_sign(x0)
    }

    #[inline(always)]
    fn exp<P: Policy>(x: Self::Vf) -> Self::Vf {
        exp_d_internal::<S, P, { EXP_MODE_EXP }>(x)
    }

    #[inline(always)]
    fn exph<P: Policy>(x: Self::Vf) -> Self::Vf {
        exp_d_internal::<S, P, { EXP_MODE_EXPH }>(x)
    }

    #[inline(always)]
    fn exp2<P: Policy>(x: Self::Vf) -> Self::Vf {
        exp_d_internal::<S, P, { EXP_MODE_POW2 }>(x)
    }

    #[inline(always)]
    fn exp10<P: Policy>(x: Self::Vf) -> Self::Vf {
        exp_d_internal::<S, P, { EXP_MODE_POW10 }>(x)
    }

    #[inline(always)]
    fn exp_m1<P: Policy>(x: Self::Vf) -> Self::Vf {
        exp_d_internal::<S, P, { EXP_MODE_EXPM1 }>(x)
    }

    #[inline(always)]
    fn cbrt<P: Policy>(x: Self::Vf) -> Self::Vf {
        let b1 = Vu64::<S>::splat(715094163); // B1 = (1023-1023/3-0.03306235651)*2**20
        let b2 = Vu64::<S>::splat(696219795); // B2 = (1023-1023/3-54/3-0.03306235651)*2**20
        let m = Vu64::<S>::splat(0x7fffffff); // u32::MAX >> 1

        let x1p54 = x * Vf64::<S>::splat(f64::from_bits(0x4350000000000000)); // 0x1p54 === 2 ^ 54

        let hx0 = (x.into_bits() >> 32) & m;

        let x_small = hx0.lt(Vu64::<S>::splat(0x00100000));

        let xs = x_small.select(x1p54, x); // note that this upcasts
        let b = x_small.select(b2, b1);

        let mut ui = xs.into_bits();
        let mut hx = (ui >> 32) & m;

        // TODO: Fix this when stable isn't broken
        hx = <Vu64<S> as Div<Divider<u64>>>::div(hx, Divider::u64(3)) + b;

        ui &= Vu64::<S>::splat(1 << 63);
        ui |= hx << 32;

        let mut t = Vf64::<S>::from_bits(ui);

        let r = (t * t) * (t / x); // encourage ILP
        let r2 = r * r;

        t *= r.poly_p::<P>(&[
            1.87595182427177009643,   /* 0x3ffe03e6, 0x0f61e692 */
            -1.88497979543377169875,  /* 0xbffe28e0, 0x92f02420 */
            1.621429720105354466140,  /* 0x3ff9f160, 0x4a49d6c2 */
            -0.758397934778766047437, /* 0xbfe844cb, 0xbee751d9 */
            0.145996192886612446982,  /* 0x3fc2b000, 0xd4e4edd7 */
        ]);

        ui = t.into_bits();
        ui = (ui + Vu64::<S>::splat(0x80000000)) & Vu64::<S>::splat(0xffffffffc0000000);
        t = Vf64::<S>::from_bits(ui);

        let r = if P::POLICY.precision >= PrecisionPolicy::Best || !S::INSTRSET.has_true_fma() {
            // original form, 5 simple ops, 2 divisions
            ((x / (t * t)) - t) / ((t + t) + (x / (t * t)))
        } else {
            // fast form, 3 simple ops, 1 division, 1 fma
            let t3 = t * t * t;
            (x - t3) / Vf64::<S>::splat(2.0).mul_add(t3, x)
        };

        t = r.mul_adde(t, t);

        if !P::POLICY.check_overflow {
            return x.eq(Vf64::<S>::zero()).select(x, t);
        }

        (hx0.gt(Vu64::<S>::splat(0x7f800000)) | hx0.eq(Vu64::<S>::zero())).select(x, t)
    }

    #[inline(always)]
    fn powf<P: Policy>(x0: Self::Vf, y: Self::Vf) -> Self::Vf {
        // define constants
        let ln2d_hi = Vf64::<S>::splat(0.693145751953125); // log(2) in extra precision, high bits
        let ln2d_lo = Vf64::<S>::splat(1.42860682030941723212E-6); // low bits of log(2)
        let log2e = Vf64::<S>::LOG2_E();
        let ln2 = Vf64::<S>::LN_2();

        // coefficients for Pade polynomials
        let zero = Vf64::<S>::zero();
        let one = Vf64::<S>::one();
        let half = Vf64::<S>::splat(0.5);

        let x1 = x0.abs();

        let mut x = fraction2::<S>(x1);

        let blend = x.gt(Vf64::<S>::splat(SQRT_2 * 0.5));

        x = x.conditional_add(x, !blend);
        x -= one;

        let x2 = x * x;

        let lg1 = (x2 * x)
            * x.poly_p::<P>(&[
                2.0039553499201281259648E1,
                5.7112963590585538103336E1,
                6.0949667980987787057556E1,
                2.9911919328553073277375E1,
                6.5787325942061044846969E0,
                4.9854102823193375972212E-1,
                4.5270000862445199635215E-5,
            ])
            / x.poly_p::<P>(&[
                6.0118660497603843919306E1,
                2.1642788614495947685003E2,
                3.0909872225312059774938E2,
                2.2176239823732856465394E2,
                8.3047565967967209469434E1,
                1.5062909083469192043167E1,
                1.0,
            ]);

        let ef = exponent_f::<S>(x1) + (blend.value() & one);

        // multiply exponent by y, nearest integer e1 goes into exponent of result, remainder yr is added to log
        let e1 = (ef * y).round();
        let yr = ef.mul_sube(y, e1); // calculate remainder yr. precision very important here

        // add initial terms to expansion
        let lg = half.nmul_adde(x2, x) + lg1; // lg = (x - 0.5f * x2) + lg1;

        // calculate rounding errors in lg
        // rounding error in multiplication 0.5*x*x
        let x2err = (half * x).mul_sube(x, half * x2);

        // rounding error in additions and subtractions
        let lgerr = half.mul_adde(x2, lg - x) - lg1; // lgerr = ((lg - x) + 0.5f * x2) - lg1;

        // extract something for the exponent
        let e2 = (lg * y * log2e).round();

        // subtract this from lg, with extra precision
        let mut v = e2.nmul_adde(ln2d_lo, lg.mul_sube(y, e2 * ln2d_hi));

        // add remainder from ef * y
        v = yr.mul_adde(ln2, v); // v += yr * VM_LN2;

        // correct for previous rounding errors
        v = (lgerr + x2err).nmul_adde(y, v); // v -= (lgerr + x2err) * y;

        // extract something for the exponent if possible
        let mut x = v;
        let e3 = (x * log2e).round();

        // high precision multiplication not needed here because abs(e3) <= 1
        x = e3.nmul_adde(ln2, x); // x -= e3 * VM_LN2;

        // poly_13m + 1, Taylor coefficients for exp function, 1/n!
        let mut z = x.poly_p::<P>(&[
            1.0, // + 1
            1.0 / 1.0,
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
        let ei = unsafe { ee.to_int_fast() };

        // biased exponent of result:
        let ej = ei + Vi64::<S>::from_bits(z.into_bits()) >> 52;

        // add exponent by integer addition
        let mut z = Vf64::<S>::from_bits((ei.into_bits() << 52) + z.into_bits());

        if !P::POLICY.check_overflow {
            return z;
        }

        // check exponent for overflow and underflow
        let overflow = Vf64::<S>::from_cast_mask(ej.ge(Vi64::<S>::splat(0x07FF))) | ee.gt(Vf64::<S>::splat(3000.0));
        let underflow = Vf64::<S>::from_cast_mask(ej.le(Vi64::<S>::splat(0x0000))) | ee.lt(Vf64::<S>::splat(-3000.0));

        // check for special cases
        let xfinite = x0.is_finite();
        let yfinite = y.is_finite();
        let efinite = ee.is_finite();

        let xzero = x0.is_zero_or_subnormal();
        let xsign = x0.is_negative();

        if thermite_unlikely!((overflow | underflow).any()) {
            z = underflow.select(zero, z);
            z = overflow.select(Vf64::<S>::infinity(), z);
        }

        let yzero = y.eq(zero);
        let yneg = y.lt(zero);

        // pow_case_x0
        z = xzero.select(yneg.select(Vf64::<S>::infinity(), yzero.select(one, zero)), z);

        let mut yodd = zero;

        if xsign.any() {
            let yint = y.eq(y.round());
            yodd = y << 63;

            let z1 = yint.select(z | yodd, x0.eq(zero).select(z, Vf64::<S>::nan()));

            yodd = yint.select(yodd, zero);

            z = xsign.select(z1, z);
        }

        let not_special = (xfinite & yfinite & (efinite | xzero));

        if thermite_likely!(not_special.all()) {
            return z; // fast return
        }

        // handle special error cases: y infinite
        let z1 = (yfinite & efinite).select(
            z,
            x1.eq(one)
                .select(one, (x1.gt(one) ^ y.is_negative()).select(Vf64::<S>::infinity(), zero)),
        );

        // handle x infinite
        let z1 = xfinite.select(
            z1,
            yzero.select(
                one,
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
    fn ln<P: Policy>(x: Self::Vf) -> Self::Vf {
        ln_d_internal::<S, P, false>(x)
    }

    #[inline(always)]
    fn ln_1p<P: Policy>(x: Self::Vf) -> Self::Vf {
        ln_d_internal::<S, P, true>(x)
    }

    #[inline(always)]
    fn log2<P: Policy>(x: Self::Vf) -> Self::Vf {
        ln_d_internal::<S, P, false>(x) * Vf64::<S>::LOG2_E()
    }

    #[inline(always)]
    fn log10<P: Policy>(x: Self::Vf) -> Self::Vf {
        ln_d_internal::<S, P, false>(x) * Vf64::<S>::LOG10_E()
    }

    #[inline(always)]
    fn erf<P: Policy>(x: Self::Vf) -> Self::Vf {
        let x2 = x * x;
        let res = x * x2.poly_rational_p::<P>(
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
            x2.gt(Vf64::<S>::splat(8.135455562428929)).select(x.signum(), res)
        } else {
            res
        }
    }

    #[inline(always)]
    fn erfinv<P: Policy>(y: Self::Vf) -> Self::Vf {
        let one = Vf64::<S>::one();

        let a = y.abs();

        let w = -a.nmul_adde(a, one).ln_p::<P>();

        // https://www.desmos.com/calculator/yduhxx1ukm values extracted via JS console
        let mut p0 = (w - Vf64::<S>::splat(2.5)).poly_p::<P>(&[
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

        let w_big = w.ge(Vf64::<S>::splat(5.0)); // at around |x| > 0.99662533231, so unlikely

        if thermite_unlikely!(w_big.any()) {
            let mut p1 = (w.sqrt() - Vf64::<S>::splat(3.0)).poly_p::<P>(&[
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
                p1 = a.eq(one).select(Vf64::<S>::infinity(), p1); // erfinv(x == 1) = inf
                p1 = a.gt(one).select(Vf64::<S>::nan(), p1); // erfinv(x > 1) = NaN
            }

            p0 = w_big.select(p1, p0);
        }

        p0 * y
    }

    #[inline(always)]
    fn next_float<P: Policy>(x: Self::Vf) -> Self::Vf {
        let i1 = Vu64::<S>::one();

        let v = x.eq(Vf64::<S>::neg_zero()).select(Vf64::<S>::zero(), x);

        let bits = v.into_bits();
        let finite = Vf64::<S>::from_bits(v.ge(Vf64::<S>::zero()).select(bits + i1, bits - i1));

        if P::POLICY.check_overflow {
            return finite;
        }

        x.eq(Vf64::<S>::infinity()).select(x, finite)
    }

    #[inline(always)]
    fn prev_float<P: Policy>(x: Self::Vf) -> Self::Vf {
        let i1 = Vu64::<S>::one();

        let v = x.eq(Vf64::<S>::zero()).select(Vf64::<S>::neg_zero(), x);

        let bits = v.into_bits();
        let finite = Vf64::<S>::from_bits(v.gt(Vf64::<S>::zero()).select(bits - i1, bits + i1));

        if P::POLICY.check_overflow {
            return finite;
        }

        x.eq(Vf64::<S>::neg_infinity()).select(x, finite)
    }
}

#[inline(always)]
fn fraction2<S: Simd>(x: Vf64<S>) -> Vf64<S> {
    // set exponent to 0 + bias
    (x & Vf64::<S>::splat(f64::from_bits(0x000FFFFFFFFFFFFF))) | Vf64::<S>::splat(f64::from_bits(0x3FE0000000000000))
}

#[inline(always)]
fn exponent<S: Simd>(x: Vf64<S>) -> Vi32<S> {
    // shift out sign, extract exp, subtract bias
    Vi32::<S>::from_bits(<Vu32<S> as SimdFromCast<S, Vu64<S>>>::from_cast(
        (x.into_bits() << 1) >> 53,
    )) - Vi32::<S>::splat(0x3FF)
}

#[inline(always)]
fn exponent_f<S: Simd>(x: Vf64<S>) -> Vf64<S> {
    let pow2_52 = Vf64::<S>::splat(4503599627370496.0);
    let bias = Vf64::<S>::splat(1023.0);

    Vf64::<S>::from_bits((x.into_bits() >> 52) | pow2_52.into_bits()) - (pow2_52 + bias)
}

#[inline(always)]
fn ln_d_internal<S: Simd, P: Policy, const P1: bool>(x0: Vf64<S>) -> Vf64<S> {
    let ln2_hi = Vf64::<S>::splat(0.693359375);
    let ln2_lo = Vf64::<S>::splat(-2.121944400546905827679E-4);
    let one = Vf64::<S>::one();
    let zero = Vf64::<S>::zero();

    let x1 = if P1 { x0 + one } else { x0 };

    let mut x = fraction2::<S>(x1);
    let mut fe = <Vf64<S> as SimdFromCast<S, Vi32<S>>>::from_cast(exponent::<S>(x1));

    let blend = x.gt(Vf64::<S>::splat(SQRT_2 * 0.5));

    x = x.conditional_add(x, !blend);
    fe = fe.conditional_add(one, blend);

    let xp1 = x - one;

    x = if P1 {
        // log(x+1). Avoid loss of precision when adding 1 and later subtracting 1 if exponent = 0
        fe.eq(zero).select(x0, xp1)
    } else {
        // log(x). Expand around 1.0
        xp1
    };

    let x2 = x * x;
    let x3 = x * x2;

    let mut res =
        x3 * x.poly_p::<P>(&[
            7.70838733755885391666E0,
            1.79368678507819816313E1,
            1.44989225341610930846E1,
            4.70579119878881725854E0,
            4.97494994976747001425E-1,
            1.01875663804580931796E-4,
        ]) / x.poly_p::<P>(&[
            2.31251620126765340583E1,
            7.11544750618563894466E1,
            8.29875266912776603211E1,
            4.52279145837532221105E1,
            1.12873587189167450590E1,
            1.0,
        ]);

    res = fe.mul_adde(ln2_lo, res); // res += fe * ln2_lo;
    res += x2.nmul_adde(Vf64::<S>::splat(0.5), x); // res += x - 0.5 * x2;
    res = fe.mul_adde(ln2_hi, res); // res += fe * ln2_hi;

    if !P::POLICY.check_overflow {
        return res;
    }

    let overflow = !x1.is_finite();
    let underflow = x1.lt(Vf64::<S>::splat(2.2250738585072014E-308));

    if thermite_likely!((overflow | underflow).none()) {
        return res;
    }

    res = underflow.select(Vf64::<S>::nan(), res); // x1 < 0 gives NAN
    res = x1.is_zero_or_subnormal().select(Vf64::<S>::neg_infinity(), res); // x1 == 0 gives -INF
    res = overflow.select(x1, res); // INF or NAN goes through
    res = (x1.is_infinite() & x1.is_negative()).select(Vf64::<S>::nan(), res); // -INF gives NAN

    res
}

#[inline(always)]
fn atan_internal<S: Simd, P: Policy, const ATAN2: bool>(y: Vf64<S>, x: Vf64<S>) -> Vf64<S> {
    let morebits = Vf64::<S>::splat(6.123233995736765886130E-17);
    let morebitso2 = Vf64::<S>::splat(6.123233995736765886130E-17 * 0.5);
    let t3po8 = Vf64::<S>::splat(SQRT_2 + 1.0);
    let neg_one = Vf64::<S>::neg_one();
    let one = Vf64::<S>::one();
    let zero = Vf64::<S>::zero();

    let mut swapxy = Mask::new(unsafe { Vf64::<S>::undefined() });

    let t = if ATAN2 {
        let x1 = x.abs();
        let y1 = y.abs();

        swapxy = y1.gt(x1);

        let mut x2 = swapxy.select(y1, x1);
        let mut y2 = swapxy.select(x1, y1);

        if P::POLICY.check_overflow {
            let both_inf = x.is_infinite() & y.is_infinite();

            // TODO: Benchmark this branch
            if thermite_unlikely!(both_inf.any()) {
                x2 = both_inf.select(x2 & neg_one, x2);
                y2 = both_inf.select(y2 & neg_one, y2);
            }
        }

        y2 / x2
    } else {
        y.abs()
    };

    let not_big = t.le(t3po8);
    let not_small = t.ge(Vf64::<S>::splat(0.66));

    let s = not_big.select(Vf64::<S>::FRAC_PI_4(), Vf64::<S>::FRAC_PI_2()) & not_small.value();

    let fac = not_big.select(morebitso2, morebits) & not_small.value();

    let a = (not_big.value() & t) + (not_small.value() & neg_one);
    let b = (not_big.value() & one) + (not_small.value() & t);

    let z = a / b;

    let zz = z * z;

    let re0 = zz.poly_p::<P>(&[
        -6.485021904942025371773E1,
        -1.228866684490136173410E2,
        -7.500855792314704667340E1,
        -1.615753718733365076637E1,
        -8.750608600031904122785E-1,
    ]) / zz.poly_p::<P>(&[
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
        re = swapxy.select(Vf64::<S>::FRAC_PI_2() - re, re);
        re = (x | y).eq(zero).select(zero, re); // atan2(0,0) = 0 by convention
                                                // also for x = -0.
        re = x.select_negative(Vf64::<S>::PI() - re, re);
    }

    re.combine_sign(y)
}

#[inline(always)]
fn asin_internal<S: Simd, P: Policy, const ACOS: bool>(x: Vf64<S>) -> Vf64<S> {
    let one = Vf64::<S>::one();

    let xa = x.abs();

    let is_big = xa.ge(Vf64::<S>::splat(0.625));

    let x1 = is_big.select(one - xa, xa * xa);

    let x2 = x1 * x1;
    let x4 = x2 * x2;
    let x8 = x4 * x4;

    let undef = unsafe { Vf64::<S>::undefined() };

    let mut px = undef;
    let mut qx = undef;
    let mut rx = undef;
    let mut sx = undef;
    let mut xb = undef;

    let bitmask = is_big.bitmask();

    // if not all are big (if any are small)
    if P::POLICY.avoid_branching || !bitmask.all() {
        px = x1.poly_p::<P>(&[
            -8.198089802484824371615E0,
            1.956261983317594739197E1,
            -1.626247967210700244449E1,
            5.444622390564711410273E0,
            -6.019598008014123785661E-1,
            4.253011369004428248960E-3,
        ]);

        qx = x1.poly_p::<P>(&[
            -4.918853881490881290097E1,
            1.395105614657485689735E2,
            -1.471791292232726029859E2,
            7.049610280856842141659E1,
            -1.474091372988853791896E1,
            1.0,
        ]);
    }

    // if any are big
    if P::POLICY.avoid_branching || bitmask.any() {
        xb = (x1 + x1).sqrt();

        rx = x1.poly_p::<P>(&[
            2.853665548261061424989E1,
            -2.556901049652824852289E1,
            6.968710824104713396794E0,
            -5.634242780008963776856E-1,
            2.967721961301243206100E-3,
        ]);

        sx = x1.poly_p::<P>(&[
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

    let frac_pi_2 = Vf64::<S>::FRAC_PI_2();

    if ACOS {
        let z1 = x.select_negative(Vf64::<S>::PI() - z1, z1);
        let z2 = frac_pi_2 - z2.combine_sign(x);
        is_big.select(z1, z2)
    } else {
        let z1 = frac_pi_2 - z1;
        is_big.select(z1, z2).combine_sign(x)
    }
}

#[inline(always)]
fn pow2n_d<S: Simd>(n: Vf64<S>) -> Vf64<S> {
    let pow2_52 = Vf64::<S>::splat(4503599627370496.0);
    let bias = Vf64::<S>::splat(1023.0);

    (n + (bias + pow2_52)) << 52
}

#[inline(always)]
fn exp_d_internal<S: Simd, P: Policy, const MODE: u8>(x0: Vf64<S>) -> Vf64<S> {
    let zero = Vf64::<S>::zero();
    let one = Vf64::<S>::one();

    let mut x = x0;
    let mut r;

    let max_x;

    match MODE {
        EXP_MODE_POW2 => {
            max_x = 1022.0;

            r = x0.round();

            x -= r;
            x *= Vf64::<S>::LN_2();
        }
        EXP_MODE_POW10 => {
            max_x = 307.65;

            let log10_2_hi = Vf64::<S>::splat(0.30102999554947019); // log10(2) in two parts
            let log10_2_lo = Vf64::<S>::splat(1.1451100899212592E-10);

            r = (x0 * Vf64::<S>::splat(LN_10 * LOG2_E)).round();

            x = r.nmul_adde(log10_2_hi, x); // x -= r * log10_2_hi;
            x = r.nmul_adde(log10_2_lo, x); // x -= r * log10_2_lo;
            x *= Vf64::<S>::LN_10();
        }
        _ => {
            max_x = if MODE == EXP_MODE_EXP { 708.39 } else { 709.7 };

            let ln2d_hi = Vf64::<S>::splat(0.693145751953125);
            let ln2d_lo = Vf64::<S>::splat(1.42860682030941723212E-6);

            r = (x0 * Vf64::<S>::splat(LOG2_E)).round();

            x = r.nmul_adde(ln2d_hi, x); // x -= r * ln2_hi;
            x = r.nmul_adde(ln2d_lo, x); // x -= r * ln2_lo;

            if MODE == EXP_MODE_EXPH {
                r -= one;
            }
        }
    }

    // Taylor coefficients, 1/n!
    // Not using minimax approximation because we prioritize precision close to x = 0
    let mut z = x.poly_p::<P>(&[
        0.0,
        1.0 / 1.0,
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

    let n2 = pow2n_d::<S>(r);

    z = match MODE {
        EXP_MODE_EXPM1 => z.mul_adde(n2, n2 - one),
        _ => z.mul_adde(n2, n2), // (z + 1.0f) * n2
    };

    if P::POLICY.check_overflow {
        let in_range = x0.abs().lt(Vf64::<S>::splat(max_x)) & x0.is_finite();

        if thermite_likely!(in_range.all()) {
            return z;
        }

        let underflow_value = if MODE == EXP_MODE_EXPM1 {
            Vf64::<S>::neg_one()
        } else {
            Vf64::<S>::zero()
        };

        r = x0.select_negative(underflow_value, Vf64::<S>::infinity());
        z = in_range.select(z, r);
        z = x0.is_nan().select(x0, z);
    }

    z
}
