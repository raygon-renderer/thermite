use crate::{
    mask::Mask,
    math::{consts::FloatConsts, policy::policies::ExtraPrecision},
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

    #[inline(always)]
    fn lerp<P: Policy>(t: Vf<Self>, a: Vf<Self>, b: Vf<Self>) -> Vf<Self> {
        if const { Self::HAS_TRUE_FMA || P::POLICY.precision.ge(PrecisionPolicy::Reference) } {
            t.mul_add(b - a, a) // Fast and accurate, if available
        } else {
            (Vf::ONE - t) * a + t * b // Accurate but slower than FMA
        }
    }

    #[inline(always)]
    fn step<P: Policy>(x: Vf<Self>, t: Vf<Self>) -> Vf<Self> {
        // bitwise AND is much faster than blendv
        x.cmp_ge(t).value() & Vf::ONE
    }

    #[inline(always)]
    fn smoothstep<P: Policy>(x: Vf<Self>, edges: Option<(Vf<Self>, Vf<Self>)>) -> Vf<Self> {
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

        let three = Vf::splat(FloatElement::from_f32(3.0));

        // NOTE: The order of the muls is important here for instruction-level parallelism.
        (t * t) * t.nmul_adde(Vf::TWO, three)
    }

    #[inline(always)]
    fn inverse_smoothstep<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        let mut t = x.nmul_adde(Vf::TWO, Vf::ONE).asin_p::<P>();

        if const { P::POLICY.precision.le(PrecisionPolicy::Medium) } {
            t *= Vf::splat(<Self::Element as FloatElement>::from_f32(1.0) / FloatElement::from_f32(3.0));
        } else {
            // exact division for higher precisions
            t /= Vf::splat(FloatElement::from_f32(3.0));
        }

        Vf::HALF - t.sin_p::<P>()
    }

    #[inline(always)]
    fn smootherstep<P: Policy>(x: Vf<Self>, edges: Option<(Vf<Self>, Vf<Self>)>) -> Vf<Self> {
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

        let six = Vf::splat(FloatElement::from_f32(6.0));
        let ten = Vf::splat(FloatElement::from_f32(10.0));
        let neg_fifteen = Vf::splat(FloatElement::from_f32(-15.0));

        (t * t * t) * x.mul_adde(six, neg_fifteen).mul_adde(x, ten)
    }

    #[inline(always)]
    fn reciprocal<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        if const { !Self::HAS_APPROX_RCP || P::POLICY.precision.gt(PrecisionPolicy::Average) } {
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
        if const { !Self::HAS_APPROX_RCP || P::POLICY.precision.gt(PrecisionPolicy::Average) } {
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
    fn invsqrt<P: Policy>(x: Vf<Self>) -> Vf<Self> {
        if const { !Self::HAS_APPROX_RSQRT || P::POLICY.precision.ge(PrecisionPolicy::Best) } {
            Vf::ONE / x.sqrt()
        } else {
            let mut y = x.rsqrt();

            if const { P::POLICY.precision.gt(PrecisionPolicy::Worst) } {
                let nx2 = Vf::splat(FloatElement::from_f32(-0.5));
                let threehalfs = Vf::splat(FloatElement::from_f32(1.5));

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
    fn ln1p<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    fn log2<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    fn log10<P: Policy>(x: Vf<Self>) -> Vf<Self>;

    fn erf<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    fn erfc<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    fn erfinv<P: Policy>(x: Vf<Self>) -> Vf<Self>;

    // fn tgamma<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    // fn lgamma<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    // fn digamma<P: Policy>(x: Vf<Self>) -> Vf<Self>;
    // fn beta<P: Policy>(x: Vf<Self>, y: Vf<Self>) -> Vf<Self>;

    #[inline(always)]
    fn gaussian<P: Policy>(x: Vf<Self>, a: Vf<Self>, c: Vf<Self>) -> Vf<Self> {
        let xc = if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
            x * c.reciprocal_p::<P>()
        } else {
            x / c
        };

        a * (Vf::splat(FloatElement::from_f32(-0.5)) * xc * xc).exp_p::<P>()
    }

    #[inline(always)]
    fn gaussian_integral<P: Policy>(x0: Vf<Self>, x1: Vf<Self>, a: Vf<Self>, c: Vf<Self>) -> Vf<Self> {
        // https://www.wolframalpha.com/input?i=integrate%20a*e%5E(-1%2F2%20*%20x%5E2%2Fc%5E2)%20from%20x%3Dx_0%20to%20x%3Dx_1
        let common = Vf::SQRT_FRAC_PI_2 * a * c;
        let denom = Vf::SQRT_2 * c;

        let (a1, a0) = if const { P::POLICY.precision.le(PrecisionPolicy::Worst) } {
            let d = denom.reciprocal_p::<ExtraPrecision<P>>();
            (x1 * d, x0 * d)
        } else {
            (x1 / denom, x0 / denom)
        };

        common * (a1.erf_p::<P>() - a0.erf_p::<P>())
    }
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
