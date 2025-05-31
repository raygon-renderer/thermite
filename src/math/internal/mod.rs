use crate::{
    Vector,
    mask::Mask,
    register::{FloatElement, FloatRegister, Register, SignedIntegerRegister},
};

use super::policy::{Policy, PolicyParameters, PrecisionPolicy};

type Vf<R> = Vector<R>;
type Vu<R> = Vector<<R as FloatRegister>::Bits>;
type Vs<R> = Vector<<R as FloatRegister>::Signed>;

pub trait MathInternal<E>: FloatRegister<Element = E> {
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
}

pub mod pd;
pub mod ps;
