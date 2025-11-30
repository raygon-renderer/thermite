use core::marker::PhantomData;

use crate::{
    mask::Mask,
    math::{Math, consts::FloatConsts, policy::policies::ExtraPrecision},
    register::{FloatElement, FloatRegister, Register, SignedIntegerRegister},
    vector::Vector,
};

use super::MathWithPolicy;
use super::policy::{Policy, PolicyParameters, PrecisionPolicy};

pub(crate) type Vf<R> = Vector<R>;
pub(crate) type Vu<R> = Vector<<R as FloatRegister>::Bits>;
pub(crate) type Vs<R> = Vector<<R as FloatRegister>::Signed>;

pub trait MathInternal<E: FloatConsts>: FloatRegister<Element = E> {
    #[inline(always)]
    fn ldexp<P: Policy>(x: Vf<Self>, exp: Vs<Self>) -> Vf<Self> {
        let bits: Vu<Self> = x.into_bits();

        // (bits >> mantissa) & mask
        let biased_exp =
            Vs::<Self>::from_bits((bits >> E::MANTISSA) & const { Vu::<Self>::splat_const(E::EXP_LSB_MASK) });

        let mut exp = biased_exp + exp; // offset exponent

        if const { P::POLICY.check_overflow } {
            // clamp exponent between 0 and MAX_BIASED_EXP
            exp = exp
                .max(Vs::<Self>::ZERO)
                .min(const { Vs::<Self>::splat_const(E::MAX_BIASED_EXP) });
        }

        let sign_mantissa = Vs::<Self>::from_bits(bits & const { Vu::<Self>::splat_const(E::SIGN_MANTISSA_MASK) });

        let mut result = (exp << E::MANTISSA) | sign_mantissa;

        if const { P::POLICY.check_overflow } {
            let is_underflow = exp.cmp_le(Vs::<Self>::ZERO);
            let input_was_subnormal = biased_exp.cmp_eq(Vs::<Self>::ZERO);

            // result = !(is_underflow | input_was_subnormal) & result
            result = (is_underflow | input_was_subnormal).value().bitandnot(result);
        }

        Vf::from_bits(result)
    }

    #[inline(always)]
    fn frexp<P: Policy>(x: Vf<Self>) -> (Vf<Self>, Vs<Self>) {
        let bits: Vu<Self> = x.into_bits();

        // (bits >> mantissa) & mask
        let biased_exp =
            Vs::<Self>::from_bits((bits >> E::MANTISSA) & const { Vu::<Self>::splat_const(E::EXP_LSB_MASK) });

        let mut exp: Vs<Self> = biased_exp - const { Vs::<Self>::splat_const(E::FREXP_BIAS_OFFSET) };

        // extract sign and mantissa, then give it the correct exponent
        let sign_mantissa: Vu<Self> = bits & const { Vu::<Self>::splat_const(E::SIGN_MANTISSA_MASK) };
        let mut fraction = sign_mantissa | const { Vu::<Self>::splat_const(E::HALF_EXP_BITS) };

        if const { P::POLICY.check_overflow } {
            // if input was zero or subnormal, set fraction to zero and exponent to zero
            let is_normal = biased_exp.cmp_ne(Vector::ZERO).value();
            exp &= is_normal;
            fraction &= Vu::<Self>::from_bits(is_normal);
        }

        (Vf::from_bits(fraction), exp)
    }

    #[inline(always)]
    fn to_degrees<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        x * Vf::FRAC_180_PI
    }

    #[inline(always)]
    fn to_radians<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        x * Vf::FRAC_PI_180
    }

    #[inline(always)]
    fn tolerance<P: Policy>() -> Vf<Self> {
        Vf::splat(E::from_i64(P::POLICY.precision.tolerance()) * E::EPSILON)
    }

    #[inline(always)]
    fn poly<P: Policy, const N: usize>(x: Vf<Self>, coeffs: &[E; N]) -> Vf<Self> {
        if const { !P::POLICY.unroll_loops || P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            // basic Horner's method that's both compact and accurate, even without FMA
            let mut res = Vf::splat(coeffs[N - 1]);
            for &c in coeffs.iter().rev().skip(1) {
                res = res.mul_adde(x, Vf::splat(c));
            }
            return res;
        }

        fast_polynomial::poly_f_n::<_, _, N>(x, |i| unsafe { Vf::splat(*coeffs.get_unchecked(i)) })
    }

    #[inline(always)]
    fn poly_rev<P: Policy, const N: usize>(x: Vf<Self>, coeffs: &[E; N]) -> Vf<Self> {
        if const { !P::POLICY.unroll_loops || P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            // basic Horner's method that's both compact and accurate, even without FMA
            let mut res = Vf::splat(coeffs[0]);
            for &c in coeffs.iter().skip(1) {
                res = res.mul_adde(x, Vf::splat(c));
            }
            return res;
        }

        fast_polynomial::poly_f_n::<_, _, N>(x, |i| unsafe { Vf::splat(*coeffs.get_unchecked(N - 1 - i)) })
    }

    #[inline(always)]
    fn poly_rational<P: Policy, const N: usize, const D: usize>(
        x: Vf<Self>,
        numerator: &[E; N],
        denominator: &[E; D],
    ) -> Vf<Self> {
        if const { P::POLICY.precision.le(PrecisionPolicy::Average) } {
            let n = Self::poly::<P, N>(x, numerator);
            let d = Self::poly::<P, D>(x, denominator);

            return n / d;
        }

        let invert = x.cmp_gt(Vf::ONE);

        let mut n0 = Vf::EMPTY;
        let mut n1 = Vf::EMPTY;
        let mut d0 = Vf::EMPTY;
        let mut d1 = Vf::EMPTY;

        if P::POLICY.avoid_branching || !invert.all() {
            n0 = Self::poly::<P, N>(x, numerator);
            d0 = Self::poly::<P, D>(x, denominator);
        }

        let mut z = Vf::EMPTY;

        if P::POLICY.avoid_branching || invert.any() {
            z = Self::reciprocal::<P>(x);
            n1 = Self::poly_rev::<P, N>(z, numerator);
            d1 = Self::poly_rev::<P, D>(z, denominator);
        }

        let n = invert.select(n1, n0);
        let d = invert.select(d1, d0);

        let res = n / d;

        // no correction needed if same degree
        if const { N == D } {
            return res;
        }

        if P::POLICY.avoid_branching || invert.any() {
            // when the degree of the numerator and denominator are different, we need to correct
            // the result by shifting over the difference in degrees
            let (mut u, mut e) = if N < D { (z, D - N) } else { (x, N - D) };

            let mut corrected = res;

            // `res = res * powi(u, e)` assuming e > 0
            // because e > 0 we can jump straight into the loop without a pre-check,
            // and avoid an extra square of u at the end
            loop {
                if e & 1 != 0 {
                    corrected *= u;
                }

                e >>= 1;

                if e == 0 {
                    // correction isn't actually needed for non-inverted case
                    return invert.select(corrected, res);
                }

                u *= u;
            }
        }

        res
    }

    fn sum_f<P: Policy, F>(start: i64, end: i64, mut f: F) -> Result<Vf<Self>, Vf<Self>>
    where
        F: FnMut(i64) -> Vf<Self>,
    {
        let mut sum = Vf::<Self>::ZERO;
        let mut c = Vf::<Self>::ZERO; // Kahan summation compensation
        let mut n = start;

        let tolerance = Self::tolerance::<P>();

        let mut converged = false;

        for _ in 0..P::POLICY.max_iterations {
            if n >= end {
                break;
            }

            let mut delta = f(n);
            let abs_delta = delta.abs();

            if abs_delta.cmp_le(tolerance).all() {
                converged = true;
                break;
            }

            let t = sum + delta;

            if P::POLICY.precision.ge(PrecisionPolicy::Best) {
                // if |sum| >= |input[i]| then
                //     c += (sum - t) + input[i] // If sum is bigger, low-order digits of input[i] are lost.
                // else
                //     c += (input[i] - t) + sum // Else low-order digits of sum are lost.
                // endif
                sum.abs().cmp_lt(abs_delta).swap(&mut sum, &mut delta);

                c += (sum - t) + delta;
            }

            sum = t;
            n += 1;
        }

        if P::POLICY.precision.ge(PrecisionPolicy::Best) {
            sum += c; // apply any remaining compensation
        }

        match converged {
            true => Ok(sum),
            false => Err(sum),
        }
    }

    fn prod_f<P: Policy, F>(start: i64, end: i64, mut f: F) -> Result<Vf<Self>, Vf<Self>>
    where
        F: FnMut(i64) -> Vf<Self>,
    {
        let mut prod = Vf::ONE;
        let mut n = start;

        let tolerance = Self::tolerance::<P>();

        for _ in 0..P::POLICY.max_iterations {
            if n >= end {
                break;
            }

            let new_prod = prod * f(n);

            let delta = new_prod - prod;

            if delta.abs().cmp_le(tolerance).all() {
                return Ok(prod);
            }

            prod = new_prod;
            n += 1;
        }

        Err(prod)
    }

    #[inline(always)]
    fn newtons_method<P: Policy, F>(
        mut x: Vf<Self>,
        tolerance: Vf<Self>,
        bounds: Option<(Vf<Self>, Vf<Self>)>,
        mut f: F,
    ) -> Vf<Self>
    where
        F: FnMut(Vf<Self>) -> (Vf<Self>, Vf<Self>),
    {
        for _ in 0..P::POLICY.max_iterations {
            let (y, y_prime) = f(x);
            let delta = y / y_prime;

            let mut stop = delta.abs().cmp_le(tolerance);

            if P::POLICY.check_overflow {
                stop |= y_prime.abs().cmp_le(tolerance);
            }

            if stop.all() {
                break;
            }

            x = stop.select(x, x - delta);

            if let Some((min, max)) = bounds {
                x = x.clamp(min, max);
            }
        }

        x
    }

    #[inline(always)]
    fn lerp<P: Policy>(t: Vf<Self>, a: Vf<Self>, b: Vf<Self>) -> Vf<Self> {
        if const { Self::HAS_TRUE_FMA || P::POLICY.precision.ge(PrecisionPolicy::Reference) } {
            t.mul_add(b - a, a) // Fast and accurate, if available
        } else {
            (Vf::ONE - t) * a + t * b // Accurate but slower than FMA
        }
    }

    #[inline(always)]
    fn scale<P: Policy>(
        x: Vf<Self>,
        in_min: Vf<Self>,
        in_max: Vf<Self>,
        out_min: Vf<Self>,
        out_max: Vf<Self>,
    ) -> Vf<Self> {
        let in_range = in_max - in_min;

        let mut t = x - in_min;

        t = if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
            t * in_range.rcp()
        } else {
            t / in_range
        };

        Self::lerp::<P>(t, out_min, out_max)
    }

    #[inline(always)]
    fn step<P: Policy>(x: Vf<Self>, t: Vf<Self>) -> Vf<Self> {
        // bitwise AND is much faster than blendv
        x.cmp_ge(t).value() & Vf::ONE
    }

    #[inline(always)]
    fn smoothstep<P: Policy, const N: usize>(x: Vf<Self>, edges: Option<(Vf<Self>, Vf<Self>)>) -> Vf<Self> {
        let mut t = x;

        if let Some((a, b)) = edges {
            let xa = t - a;
            let ba = b - a;

            t = if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
                xa * ba.rcp()
            } else {
                xa / ba
            };
        }

        if P::POLICY.check_overflow {
            t = t.clamp(Vf::ZERO, Vf::ONE);
        }

        match N {
            // t was already scaled to between the edges
            0 => Self::step::<P>(t, Vf::HALF),
            1 => t, // linear
            _ => {
                t.powi(N as i32)
                    * const { Smoothstep::<E, N>::COEFFICIENTS }
                        .into_iter()
                        .fold(Vf::ZERO, |res, c| res.mul_adde(t, Vf::splat(E::from_i64(c))))
            }
        }
    }

    #[inline(always)]
    fn smoothstep_derivative<P: Policy, const N: usize>(x: Vf<Self>, edges: Option<(Vf<Self>, Vf<Self>)>) -> Vf<Self> {
        let mut t = x;
        let mut dt_dx = Vf::ONE;

        if let Some((a, b)) = edges {
            let xa = t - a;
            let ba = b - a;

            (dt_dx, t) = if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
                let bar = ba.rcp();

                (bar, xa * bar)
            } else {
                (Vf::ONE / ba, xa / ba)
            };
        }

        match N {
            // derivative of step function is infinite at 0.5, so-called Dirac delta function
            0 => t.cmp_eq(Vf::HALF).select(Vf::INFINITY, Vf::ZERO),
            1 => dt_dx,
            _ => {
                if P::POLICY.check_overflow {
                    t = t.clamp(Vf::ZERO, Vf::ONE);
                }

                let y =
                    const { Smoothstep::<E, N>::COEFFICIENTS }
                        .into_iter()
                        .enumerate()
                        .fold(Vf::ZERO, |res, (k, c)| {
                            // order - k for derivative coefficient
                            res.mul_adde(t, Vf::splat(E::from_i64(c) * E::from_i64((2 * N - k - 1) as i64)))
                        });

                y * dt_dx * t.powi((N - 1) as i32)
            }
        }
    }

    #[inline(always)]
    fn inverse_smoothstep<P: Policy, const N: usize>(y: Vf<Self>, edges: Option<(Vf<Self>, Vf<Self>)>) -> Vf<Self> {
        let mut ba = Vf::ONE;
        let mut bar = Vf::ONE;
        let mut bar_a = Vf::ONE; // (b - a) * a

        // Start with an initial guess of 0.5, since that'll have the largest derivative
        let mut x0 = Vf::HALF;

        if let Some((a, b)) = edges {
            ba = b - a;

            if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
                bar = ba.rcp();
                bar_a = bar * a;
            } else {
                bar = Vf::ONE / ba;
                bar_a = a / ba;
            }

            match N {
                0 => return y.step_p::<P>(Vf::HALF).mul_adde(ba, a),
                1 => return y.mul_adde(ba, a),

                // scale the initial guess to fit the edges
                _ => x0 = x0.mul_adde(ba, a),
            }
        }

        match N {
            0 => return y.step_p::<P>(Vf::HALF),
            1 => return y,

            // N=2 has a closed-form solution
            2 => {
                let mut t = y.nmul_adde(Vf::TWO, Vf::ONE).asin_p::<P>();

                if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
                    t *= Vf::splat(E::ONE / E::from_f64(3.0));
                } else {
                    // exact division for higher precisions
                    t /= Vf::splat(E::from_f64(3.0));
                }

                t = Vf::HALF - t.sin_p::<P>();

                if let Some((a, _)) = edges {
                    // rescale to original edges
                    t = t.mul_adde(ba, a);
                }

                return t;
            }
            _ => {}
        }

        let bounds = edges.or(Some((Vf::ZERO, Vf::ONE)));

        Self::newtons_method::<P, _>(x0, Self::tolerance::<P>(), bounds, |x: Vf<Self>| {
            let mut t = x;
            let mut dt_dx = bar;

            if edges.is_some() {
                // adjust by precalculated scales
                t = t.mul_sube(bar, bar_a);
            }

            let xn1 = t.powi((N - 1) as i32);

            #[rustfmt::skip]
            let (fx, fpx) = const { Smoothstep::<E, N>::COEFFICIENTS }.into_iter().enumerate().fold(
                (Vf::ZERO, Vf::ZERO),
                |(fx, fpx), (k, c)| {(
                    fx.mul_adde(t, Vf::splat(E::from_i64(c))),
                    fpx.mul_adde(t, Vf::splat(E::from_i64(c) * E::from_i64((2 * N - k - 1) as i64))),
                )},
            );

            (t.mul_sube(xn1 * fx, y), fpx * dt_dx * xn1)
        })
    }

    #[inline(always)]
    fn smooth_interpolator<P: Policy>(x: Vf<Self>, edges: Option<(Vf<Self>, Vf<Self>)>, k: Vf<Self>) -> Vf<Self> {
        let mut t = x;

        if let Some((a, b)) = edges {
            // rescale t to [0, 1]
            t = (t - a) / (b - a);
        }

        let kt = k * t;

        // (2x-1) / (kx^2-kx)
        let e = t.mul_sube(Vf::TWO, Vf::ONE) / kt.mul_sube(t, kt);

        // exp(e) + 1
        let d = e.exp_p::<P>() + Vf::ONE;

        // 1/(exp(e) + 1), it's important this is done in extra precision
        let mut res = d.reciprocal_p::<ExtraPrecision<P>>();

        let overflow = e.is_infinite();

        // If the denominator is small enough, it could cause overflow,
        // however that only really happens when t is very close to 0 or 1,
        // or when k is very small. So approximate it with a step function.
        if P::POLICY.avoid_branching || crate::unlikely(overflow.any()) {
            res = overflow.select(t.step_p::<P>(Vf::HALF), res);
        }

        // these are important since the Exp formulation is discontinuous at 0 and 1,
        // and this maintains the asymptotes when t is outside the range [0, 1]
        res = t.cmp_ge(Vf::ONE).select(Vf::ONE, res);
        res = t.cmp_le(Vf::ZERO).select(Vf::ZERO, res);

        res
    }

    #[inline(always)]
    fn smooth_interpolator_inverse<P: Policy>(
        mut y: Vf<Self>,
        edges: Option<(Vf<Self>, Vf<Self>)>,
        k: Vf<Self>,
    ) -> Vf<Self> {
        // k ln(1/y - 1)
        let l = k * (y.reciprocal_p::<P>() - Vf::ONE).ln_p::<P>();

        // ((l + 2) - sqrt(l^2 + 4)) / 2l
        let a = (l + Vf::TWO);
        let b = l.mul_adde(l, Vf::splat(E::from_i64(4))).sqrt();
        let mut t = (a - b) / (Vf::TWO * l);

        // handle out-of-bounds inputs
        t = y.cmp_ge(Vf::ONE).select(Vf::ONE, t);
        t = y.cmp_le(Vf::ZERO).select(Vf::ZERO, t);

        if let Some((a, b)) = edges {
            // rescale t to the original edges
            t = t.mul_adde(b - a, a);
        }

        t
    }

    #[inline(always)]
    fn reciprocal<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        if const { !Self::HAS_APPROX_RCP || P::POLICY.precision.ge(PrecisionPolicy::Average) } {
            Vf::ONE / x
        } else {
            let mut y = x.rcp();

            if const { P::POLICY.precision.gt(PrecisionPolicy::Worst) } {
                // one iteration of Newton's method
                y = y * x.nmul_adde(y, Vf::TWO);
            }

            y
        }
    }

    #[inline(always)]
    fn reciprocal_adde<P: Policy>(x: Vf<Self>, a: Vf<Self>) -> Vf<Self> {
        if const { !Self::HAS_APPROX_RCP || P::POLICY.precision.ge(PrecisionPolicy::Average) } {
            a + Vf::ONE / x
        } else {
            let mut y = x.rcp();

            if const { P::POLICY.precision.gt(PrecisionPolicy::Worst) } {
                // one iteration of Newton's method
                y = y.mul_adde(x.nmul_adde(y, Vf::TWO), a);
            } else {
                y += a;
            }

            y
        }
    }

    #[inline(always)]
    fn inverse_sqrt<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        if const { !Self::HAS_APPROX_RSQRT || P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            Vf::ONE / x.sqrt()
        } else {
            let mut y = x.rsqrt();

            if const { P::POLICY.precision.gt(PrecisionPolicy::Worst) } {
                let nx2 = Vf::splat(E::from_f64(-0.5));
                let threehalfs = Vf::splat(E::from_f64(1.5));

                // one iteration of Newton's method
                y = y * (y * y).mul_adde(nx2, threehalfs);
            }

            y
        }
    }

    #[inline(always)]
    fn powi<P: Policy>(mut x: Vf<Self>, mut e: i32) -> Vf<Self> {
        let mut res = Vf::ONE;

        if e < 0 {
            e = -e;
            x = Self::reciprocal::<P>(x);
        }

        while e != 0 {
            if e & 1 != 0 {
                res *= x;
            }

            x *= x;
            e >>= 1;
        }

        res
    }

    #[inline(always)]
    fn powiv<P: Policy>(mut x: Vf<Self>, mut e: Vs<Self>) -> Vf<Self> {
        let mut res = Vf::ONE;

        x = e.is_negative().select(Self::reciprocal::<P>(x), x);
        e = e.abs();

        loop {
            let mut e1 = e & Vs::<Self>::ONE;

            let nx = res * x;

            // NOTE: e1 is bitcast to Self when `select` is used, so we use it for the MSB_BLENDV hack
            // requirements
            res = if <Self as Register>::HAS_MSB_BLENDV {
                // Move the lowest bit to the highest bit position
                e1 <<= const { core::mem::size_of::<<Self::Signed as Register>::Element>() as u32 * 8 - 1 };

                // Blend the result based on the highest bit of e1
                Mask::from_unchecked(e1).select(nx, res)
            } else {
                e1.cmp_ne(Vs::<Self>::ZERO).select(nx, res)
            };

            x *= x;
            e >>= 1;

            if e.cmp_ne(Vs::<Self>::ZERO).none() {
                return res;
            }
        }

        res
    }

    #[inline(always)]
    fn hypot<P: Policy>(x: Vf<Self>, y: Vf<Self>) -> Vf<Self> {
        if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
            // Use the worst precision method, which is usually faster
            x.mul_adde(x, y * y).sqrt()
        } else {
            // Use a more precise method
            let x = x.abs();
            let y = y.abs();

            let max = x.max(y);
            let min = x.min(y);
            let t = min / max;

            let mut res = max * t.mul_adde(t, Vf::ONE).sqrt();

            if P::POLICY.check_overflow {
                // because these have already been abs, we can just use less-than
                let inf = Vf::INFINITY;
                res = (x.cmp_lt(inf) & y.cmp_lt(inf) & t.cmp_lt(inf)).select(res, x + y);
            }

            res
        }
    }

    fn sin_cos<P: Policy>(x: Vf<Self>) -> (Vf<Self>, Vf<Self>);

    #[inline(always)]
    fn sin<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        Self::sin_cos::<P>(x).0
    }

    #[inline(always)]
    fn cos<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        Self::sin_cos::<P>(x).1
    }

    #[inline(always)]
    fn tan<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        let (s, c) = Self::sin_cos::<P>(x);
        s / c
    }

    #[inline(always)]
    fn sinc<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        let n = x.sin_p::<P>();

        let mut y = if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
            n * x.reciprocal_p::<P>()
        } else {
            n / x
        };

        if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            let x2 = x * x;
            let x4 = x2 * x2;

            let mut small_res = Vf::ONE;

            // Taylor series expansion for small x
            small_res -= x2 / Vf::splat(E::from_f64(6.0));
            small_res += x4 / Vf::splat(E::from_f64(120.0));

            let is_small = x.abs().cmp_le(Vf::FOURTH_ROOT_EPSILON);

            // NOTE: Taylor series is naturally 1 at x = 0, so we can use it directly
            y = is_small.select(small_res, y);
        } else {
            // Otherwise we check for zero exactly
            y = x.cmp_eq(Vf::ZERO).select(Vf::ONE, y);
        }

        if P::POLICY.check_overflow {
            y = x.is_infinite().select(Vf::ZERO, y);
        }

        y
    }

    #[inline(always)]
    fn sin_pix<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        let (x, xs) = if const { P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            let x = x.abs();
            let mut fl = x.floor();

            let is_odd = (fl % Vf::TWO).cmp_ne(Vf::ZERO);

            fl += Vf::ONE & is_odd.value(); // only add one if odd

            let sign = Vf::NEG_ZERO & is_odd.value();
            let mut dist = (x - fl) ^ sign; // flip the sign if odd

            dist -= Vf::ONE & dist.cmp_gt(Vf::HALF).value(); // if dist > 0.5, flip the sign

            (x ^ sign, dist)
        } else {
            (x, x)
        };

        x * (xs * Vf::PI).sin_p::<P>()
    }

    fn sinh<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    fn cosh<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    fn tanh<P: Policy>(x: Vf<Self>) -> Vf<Self>;

    fn asin<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    fn acos<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    fn atan<P: Policy>(y: Vf<Self>) -> Vf<Self>;
    fn atan2<P: Policy>(y: Vf<Self>, x: Vf<Self>) -> Vf<Self>;

    fn asinh<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    fn acosh<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    fn atanh<P: Policy>(x: Vf<Self>) -> Vf<Self>;

    fn exp<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    fn exph<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    fn exp2<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    fn exp10<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    fn exp_m1<P: Policy>(x: Vf<Self>) -> Vf<Self>;

    fn powf<P: Policy>(x: Vf<Self>, e: Vf<Self>) -> Vf<Self>;
    fn cbrt<P: Policy>(x: Vf<Self>) -> Vf<Self>;

    fn ln<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    fn ln_1p<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    fn log2<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    fn log10<P: Policy>(x: Vf<Self>) -> Vf<Self>;

    /// log with arbitrary base N
    #[inline(always)]
    fn log_n<P: Policy, const N: usize>(x: Vf<Self>) -> Vf<Self> {
        match N {
            // 0 and 1 are special cases, and these are what Wolfram Alpha returns
            0 => Vf::ZERO,     // log(x)/log(0) = log(x)/-infinity = 0
            1 => Vf::INFINITY, // log(x)/log(1) = log(x)/0 = complex infinity, only return real part
            2 => Self::log2::<P>(x),
            10 => Self::log10::<P>(x),
            n if n <= 32 => {
                #[rustfmt::skip] #[allow(clippy::approx_constant)]
                const LOG_TABLE: [f64; 30] = [ // precomputed 1/Table[log(x), {x, 3, 32}]
                    1.0 / 1.0986122886681096913952452369225257046474905578227, 1.0 / 1.3862943611198906188344642429163531361510002687205,
                    1.0 / 1.6094379124341003746007593332261876395256013542685, 1.0 / 1.7917594692280550008124773583807022727229906921830,
                    1.0 / 1.9459101490553133051053527434431797296370847295819, 1.0 / 2.0794415416798359282516963643745297042265004030808,
                    1.0 / 2.1972245773362193827904904738450514092949811156455, 1.0 / 2.3025850929940456840179914546843642076011014886288,
                    1.0 / 2.3978952727983705440619435779651292998217068539374, 1.0 / 2.4849066497880003102297094798388788407984908265433,
                    1.0 / 2.5649493574615367360534874415653186048052679447602, 1.0 / 2.6390573296152586145225848649013562977125848639421,
                    1.0 / 2.7080502011022100659960045701487133441730919120913, 1.0 / 2.7725887222397812376689284858327062723020005374410,
                    1.0 / 2.8332133440562160802495346178731265355882030125857, 1.0 / 2.8903717578961646922077225953032279773704812500058,
                    1.0 / 2.9444389791664404600090274318878535372373792612991, 1.0 / 2.9957322735539909934352235761425407756766016229890,
                    1.0 / 3.0445224377234229965005979803657054342845752874046, 1.0 / 3.0910424533583158534791756994233058678972069882977,
                    1.0 / 3.1354942159291496908067528318101961184423803148404, 1.0 / 3.1780538303479456196469416012970554088739909609035,
                    1.0 / 3.2188758248682007492015186664523752790512027085370, 1.0 / 3.2580965380214820454707195630234951728807680791205,
                    1.0 / 3.2958368660043290741857357107675771139424716734682, 1.0 / 3.3322045101752039239398169863595328657880849983024,
                    1.0 / 3.3672958299864740271832720323619116054945129139227, 1.0 / 3.4011973816621553754132366916068899122485920464515,
                    1.0 / 3.4339872044851462459291643245423572104499389304806, 1.0 / 3.4657359027997265470861606072908828403775006718013,
                ];

                Self::ln::<P>(x) * Vf::splat(E::from_f64(LOG_TABLE[n - 3]))
            }
            _ => Self::ln::<P>(x) / Vf::splat(E::from_f64(libm::log(N as f64))),
        }
    }

    #[inline(always)]
    fn log<P: Policy>(x: Vf<Self>, base: Vf<Self>) -> Vf<Self> {
        Self::ln::<P>(x) / Self::ln::<P>(base)
    }

    /// ln(1 - e^(-x))
    #[inline(always)]
    fn ln1m_expnx<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        (Vf::ONE - (-x).exp_p::<P>()).ln_p::<P>()
    }

    fn ln1m_expnx_ext<P: Policy>(x: Vf<Self>, lnx: Vf<Self>) -> Vf<Self>;

    fn erf<P: Policy>(x: Vf<Self>) -> Vf<Self>;

    #[inline(always)]
    fn erfc<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        Vf::ONE - Self::erf::<P>(x) // erfc(x) = 1 - erf(x), fallback implementation
    }

    fn erfinv<P: Policy>(x: Vf<Self>) -> Vf<Self>;
}

pub mod pd;
pub mod ps;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum ExpMode {
    Exp = 0,
    Expm1,
    Exph,
    Pow2,
    Pow10,
}

const EXP_MODE_EXP: u8 = ExpMode::Exp as u8;
const EXP_MODE_EXPM1: u8 = ExpMode::Expm1 as u8;
const EXP_MODE_EXPH: u8 = ExpMode::Exph as u8;
const EXP_MODE_POW2: u8 = ExpMode::Pow2 as u8;
const EXP_MODE_POW10: u8 = ExpMode::Pow10 as u8;

const fn binomial(a: i32, b: i32) -> i64 {
    if b <= 0 {
        return 1;
    }

    let mut res: i64 = 1;
    let mut i = 0;

    while i < b {
        let n: i64 = res * (a - i) as i64;
        res = n / (i + 1) as i64;

        i += 1;
    }

    res
}

struct Smoothstep<F: FloatElement, const N: usize>(PhantomData<[F; N]>);
impl<F: FloatElement, const N: usize> Smoothstep<F, N> {
    // ensure these coefficients are generated at compile time
    const COEFFICIENTS: [i64; N] = const {
        let mut coeffs = [0; N];
        let n = (N - 1) as i32;

        let mut k = 0;
        while k < N {
            let c = binomial(-1 - n, k as i32) * binomial(n + n + 1, n - k as i32);

            if c.unsigned_abs() > F::MAX_U64 {
                panic!("Binomial coefficient overflow");
            }

            // store in reverse order for easier polynomial evaluation
            coeffs[N - k - 1] = c;
            k += 1;
        }

        coeffs
    };
}
