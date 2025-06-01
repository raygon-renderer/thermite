use crate::{
    Vector,
    mask::Mask,
    register::{FloatElement, FloatRegister, Register, SignedIntegerRegister},
};

use super::policy::{Policy, PolicyParameters, PrecisionPolicy};

pub(crate) type Vf<R> = Vector<R>;
pub(crate) type Vu<R> = Vector<<R as FloatRegister>::Bits>;
pub(crate) type Vs<R> = Vector<<R as FloatRegister>::Signed>;

pub trait MathInternal<E>: FloatRegister<Element = E> {
    #[inline(always)]
    fn poly<P: Policy, const N: usize>(x: Vf<Self>, coeffs: &[E; N]) -> Vf<Self> {
        if const { !P::POLICY.unroll_loops || P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            // basic Horner's method that's both compact and accurate, even without FMA
            let mut res = Vf::<Self>::splat(coeffs[N - 1]);
            for &c in coeffs.iter().rev().skip(1) {
                res = res.mul_adde(x, Vf::<Self>::splat(c));
            }
            return res;
        }

        fast_polynomial::poly_f_n::<_, _, N>(x, |i| unsafe { Vf::<Self>::splat(*coeffs.get_unchecked(i)) })
    }

    #[inline(always)]
    fn poly_rev<P: Policy, const N: usize>(x: Vf<Self>, coeffs: &[E; N]) -> Vf<Self> {
        if const { !P::POLICY.unroll_loops || P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            // basic Horner's method that's both compact and accurate, even without FMA
            let mut res = Vf::<Self>::splat(coeffs[0]);
            for &c in coeffs.iter().skip(1) {
                res = res.mul_adde(x, Vf::<Self>::splat(c));
            }
            return res;
        }

        fast_polynomial::poly_f_n::<_, _, N>(x, |i| unsafe { Vf::<Self>::splat(*coeffs.get_unchecked(N - 1 - i)) })
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

        let invert = x.cmp_gt(Vf::<Self>::ONE);

        let mut n0 = Vf::<Self>::EMPTY;
        let mut n1 = Vf::<Self>::EMPTY;

        let mut d0 = Vf::<Self>::EMPTY;
        let mut d1 = Vf::<Self>::EMPTY;

        if P::POLICY.avoid_branching || !invert.all() {
            n0 = Self::poly::<P, N>(x, numerator);
            d0 = Self::poly::<P, D>(x, denominator);
        }

        let mut z = Vf::<Self>::EMPTY;

        if P::POLICY.avoid_branching || invert.any() {
            z = Self::reciprocal::<P>(x);
            n1 = Self::poly_rev::<P, N>(z, numerator);
            d1 = Self::poly_rev::<P, D>(z, denominator);
        }

        let n = invert.select(n0, n1);
        let d = invert.select(d0, d1);

        let res = n / d;

        // no correction needed if same degree
        if N == D {
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
                    return invert.select(res, corrected);
                }

                u *= u;
            }
        }

        res
    }

    #[inline(always)]
    fn lerp<P: Policy>(t: Vf<Self>, a: Vf<Self>, b: Vf<Self>) -> Vf<Self> {
        if const { Self::HAS_TRUE_FMA || P::POLICY.precision.ge(PrecisionPolicy::Reference) } {
            t.mul_add(b - a, a) // Fast and accurate, if available
        } else {
            (Vf::<Self>::ONE - t) * a + t * b // Accurate but slower than FMA
        }
    }

    #[inline(always)]
    fn reciprocal<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        if const { !Self::HAS_APPROX_RCP || P::POLICY.precision.gt(PrecisionPolicy::Average) } {
            Vf::<Self>::ONE / x
        } else {
            let mut y = x.rcp();

            if const { P::POLICY.precision.gt(PrecisionPolicy::Worst) } {
                // one iteration of Newton's method
                y = y * x.nmul_adde(y, Vf::<Self>::TWO);
            }

            y
        }
    }

    #[inline(always)]
    fn invsqrt<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        if const { !Self::HAS_APPROX_RSQRT || P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            Vf::<Self>::ONE / x.sqrt()
        } else {
            let mut y = x.rsqrt();

            if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
                return y;
            }

            let nx2 = Vf::<Self>::splat(FloatElement::from_f32(-0.5));
            let threehalfs = Vf::<Self>::splat(FloatElement::from_f32(1.5));

            // one iteration of Newton's method
            y = y * (y * y).mul_adde(nx2, threehalfs);

            y
        }
    }

    #[inline(always)]
    fn powi<P: Policy>(mut x: Vf<Self>, mut e: i32) -> Vf<Self> {
        let mut res = Vf::<Self>::ONE;

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
        let mut res = Vf::<Self>::ONE;

        x = e.is_negative().select(x, Self::reciprocal::<P>(x));
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
                Mask::from_unchecked(e1).select(res, nx)
            } else {
                e1.cmp_ne(Vs::<Self>::ZERO).select(res, nx)
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

            let mut res = max * t.mul_adde(t, Vf::<Self>::ONE).sqrt();

            if P::POLICY.check_overflow {
                // because these have already been abs, we can just use less-than
                let inf = Vf::<Self>::INFINITY;
                res = (x.cmp_lt(inf) & y.cmp_lt(inf) & t.cmp_lt(inf)).select(x + y, res);
            }

            res
        }
    }

    fn sincos<P: Policy>(x: Vf<Self>) -> (Vf<Self>, Vf<Self>);

    #[inline(always)]
    fn sin<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        Self::sincos::<P>(x).0
    }

    #[inline(always)]
    fn cos<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        Self::sincos::<P>(x).1
    }

    #[inline(always)]
    fn tan<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        let (s, c) = Self::sincos::<P>(x);
        s / c
    }

    fn sinh<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    fn cosh<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    fn tanh<P: Policy>(x: Vf<Self>) -> Vf<Self>;

    fn asin<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    fn acos<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    fn atan<P: Policy>(x: Vf<Self>) -> Vf<Self>;
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
    fn ln1p<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    fn log2<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    fn log10<P: Policy>(x: Vf<Self>) -> Vf<Self>;

    fn erf<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    fn erfc<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    fn erfinv<P: Policy>(x: Vf<Self>) -> Vf<Self>;
}

pub mod pd;
pub mod ps;
