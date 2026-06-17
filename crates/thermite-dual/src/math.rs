//! Transcendental math for [`Dual`] via the chain rule.
//!
//! Rather than provide inherent methods, this implements Thermite's
//! `Specialized*Math` traits for `Dual<V, N>`. The blanket impls in
//! `thermite::math` then give `Dual` vectors the full
//! [`CoreMath`](thermite::math::CoreMath) /
//! [`TranscendentalMath`](thermite::math::TranscendentalMath) /
//! [`SpatialMath`](thermite::math::SpatialMath) /
//! [`RealMath`](thermite::math::RealMath) APIs (and their `_p` policy variants)
//! for free.
//!
//! Each primitive computes the primal with the inner vector's policy math, then
//! propagates derivatives with the chain rule `f(a + bε) = f(a) + b·f'(a)ε`.
//! Every *non*-primitive (e.g. `tan`, `sin`, `powi`, `lerp`, `smoothstep`)
//! comes from the trait defaults, which compose out of the dual arithmetic and
//! are therefore differentiated automatically.

use thermite::math::FloatConsts;
use thermite::math::RealMathWithPolicy;
use thermite::math::policy::Policy;
use thermite::math::specialized::{
    SpecializedCoreMath, SpecializedRealMath, SpecializedSpatialMath, SpecializedTranscendentalMath,
};
use thermite::prelude::*;

use crate::Dual;
use crate::vector::DualFloatVector;

/// Inner vector requirements for the dual math library: a real float vector that
/// supports the full policy-aware math suite and float constants.
pub trait DualMathVector: DualFloatVector + RealMathWithPolicy + FloatConsts {}
impl<V> DualMathVector for V where V: DualFloatVector + RealMathWithPolicy + FloatConsts {}

impl<V: DualMathVector, const N: usize> SpecializedCoreMath<Dual<V::Element, N>> for Dual<V, N> {
    #[inline(always)]
    fn inverse_sqrt<P: Policy>(self) -> Self {
        let r = self.re.inverse_sqrt_p::<P>();
        // d/dx x^(-1/2) = -1/2 x^(-3/2) = -1/2 * r^3
        self.chain(r, (V::HALF * r * r * r).neg())
    }
}

impl<V: DualMathVector, const N: usize> SpecializedTranscendentalMath<Dual<V::Element, N>> for Dual<V, N> {
    #[inline(always)]
    fn sin_cos<P: Policy>(self) -> (Self, Self) {
        let (s, c) = self.re.sin_cos_p::<P>();
        (self.chain(s, c), self.chain(c, s.neg()))
    }

    // Override the `tan` default (which would be dual `sin_cos` followed by a dual
    // division). Going through the inner `tan` primitive plus the chain rule
    // `d/dx tan = 1 + tan^2` is cheaper -- it uses the element type's dedicated
    // `tan` (f32) and avoids the per-component division.
    #[inline(always)]
    fn tan<P: Policy>(self) -> Self {
        let t = self.re.tan_p::<P>();
        self.chain(t, t.mul_adde(t, V::ONE))
    }

    // The default computes `sin_cos(self * PI)` -- the generic sine/cosine of x*pi
    // rather than the dedicated `sincos_pi` primitive, which is more precise at
    // multiples of pi. Derivatives: d/dx sin(pi x) = pi cos(pi x),
    // d/dx cos(pi x) = -pi sin(pi x). `sin_pi`/`cos_pi` inherit this via the default.
    #[inline(always)]
    fn sincos_pi<P: Policy>(self) -> (Self, Self) {
        let (s, c) = self.re.sincos_pi_p::<P>();
        let pi = V::PI;
        (self.chain(s, pi * c), self.chain(c, (pi * s).neg()))
    }

    // Same reasoning as `tan`: the default is `sincos_pi` + dual division.
    // d/dx tan(pi x) = pi (1 + tan^2(pi x)).
    #[inline(always)]
    fn tan_pi<P: Policy>(self) -> Self {
        let t = self.re.tan_pi_p::<P>();
        self.chain(t, V::PI * t.mul_adde(t, V::ONE))
    }

    #[inline(always)]
    fn sinh_cosh<P: Policy>(self) -> (Self, Self) {
        let (sh, ch) = self.re.sinh_cosh_p::<P>();
        (self.chain(sh, ch), self.chain(ch, sh))
    }

    #[inline(always)]
    fn tanh<P: Policy>(self) -> Self {
        let v = self.re.tanh_p::<P>();
        // d/dx tanh = 1 - tanh^2
        self.chain(v, v.nmul_adde(v, V::ONE))
    }

    #[inline(always)]
    fn asin<P: Policy>(self) -> Self {
        let v = self.re.asin_p::<P>();
        // 1 / sqrt(1 - x^2)
        self.chain(v, self.re.nmul_adde(self.re, V::ONE).sqrt().reciprocal_p::<P>())
    }

    #[inline(always)]
    fn acos<P: Policy>(self) -> Self {
        let v = self.re.acos_p::<P>();
        // -1 / sqrt(1 - x^2)
        self.chain(v, self.re.nmul_adde(self.re, V::ONE).sqrt().reciprocal_p::<P>().neg())
    }

    #[inline(always)]
    fn atan<P: Policy>(self) -> Self {
        let v = self.re.atan_p::<P>();
        // 1 / (1 + x^2)
        self.chain(v, self.re.mul_adde(self.re, V::ONE).reciprocal_p::<P>())
    }

    #[inline(always)]
    fn asinh<P: Policy>(self) -> Self {
        let v = self.re.asinh_p::<P>();
        // 1 / sqrt(x^2 + 1)
        self.chain(v, self.re.mul_adde(self.re, V::ONE).sqrt().reciprocal_p::<P>())
    }

    #[inline(always)]
    fn acosh<P: Policy>(self) -> Self {
        let v = self.re.acosh_p::<P>();
        // 1 / sqrt(x^2 - 1)
        self.chain(v, self.re.mul_sube(self.re, V::ONE).sqrt().reciprocal_p::<P>())
    }

    #[inline(always)]
    fn atanh<P: Policy>(self) -> Self {
        let v = self.re.atanh_p::<P>();
        // 1 / (1 - x^2)
        self.chain(v, self.re.nmul_adde(self.re, V::ONE).reciprocal_p::<P>())
    }

    #[inline(always)]
    fn exp<P: Policy>(self) -> Self {
        let v = self.re.exp_p::<P>();
        self.chain(v, v)
    }

    #[inline(always)]
    fn exph<P: Policy>(self) -> Self {
        // exph(x) = 0.5 e^x, derivative = exph(x)
        let v = self.re.exph_p::<P>();
        self.chain(v, v)
    }

    #[inline(always)]
    fn exp2<P: Policy>(self) -> Self {
        let v = self.re.exp2_p::<P>();
        // d/dx 2^x = ln(2) 2^x
        self.chain(v, V::LN_2 * v)
    }

    #[inline(always)]
    fn exp10<P: Policy>(self) -> Self {
        let v = self.re.exp10_p::<P>();
        // d/dx 10^x = ln(10) 10^x
        self.chain(v, V::LN_10 * v)
    }

    #[inline(always)]
    fn exp_m1<P: Policy>(self) -> Self {
        let v = self.re.exp_m1_p::<P>();
        // d/dx (e^x - 1) = e^x = v + 1
        self.chain(v, v + V::ONE)
    }

    #[inline(always)]
    fn powf<P: Policy>(self, e: Self) -> Self {
        let v = self.re.powf_p::<P>(e.re);
        // d/dx x^y = y x^(y-1) = y * (x^y) / x = e.re * v / x;  d/dy x^y = x^y ln x
        let a = e.re * v / self.re;
        let b = v * self.re.ln_p::<P>();
        let mut dual = self.dual;
        let mut i = 0;
        while i < N {
            // a*x' + b*y'  =  fma(a, x', b*y')
            dual[i] = a.mul_adde(self.dual[i], b * e.dual[i]);
            i += 1;
        }
        Dual { re: v, dual }
    }

    #[inline(always)]
    fn cbrt<P: Policy>(self) -> Self {
        let v = self.re.cbrt_p::<P>();
        // d/dx x^(1/3) = 1 / (3 x^(2/3)) = 1 / (3 v^2)
        let three: V = thermite::const_splat!(int <V::Element>: 3);
        self.chain(v, (three * v * v).reciprocal_p::<P>())
    }

    // The default `nth_root` runs a dual `powf` plus a Halley iteration with dual
    // division -- very expensive. Use the inner dedicated `nth_root` for the value
    // and the chain rule: d/dx x^(1/M) = (1/M) x^(1/M - 1) = v / (M*x).
    #[inline(always)]
    fn nth_root<P: Policy, const M: usize>(self) -> Self {
        let v = self.re.nth_root_p::<P, M>();
        let m_v = V::splat(<V::Element as FloatElement>::from_int(M as thermite::LargeInt));
        self.chain(v, v / (m_v * self.re))
    }

    #[inline(always)]
    fn ln<P: Policy>(self) -> Self {
        let v = self.re.ln_p::<P>();
        self.chain(v, self.re.reciprocal_p::<P>())
    }

    #[inline(always)]
    fn ln_1p<P: Policy>(self) -> Self {
        let v = self.re.ln_1p_p::<P>();
        // 1 / (1 + x)
        self.chain(v, (V::ONE + self.re).reciprocal_p::<P>())
    }

    #[inline(always)]
    fn log2<P: Policy>(self) -> Self {
        let v = self.re.log2_p::<P>();
        // 1 / (x ln 2)
        self.chain(v, self.re.reciprocal_p::<P>() / V::LN_2)
    }

    #[inline(always)]
    fn log10<P: Policy>(self) -> Self {
        let v = self.re.log10_p::<P>();
        // 1 / (x ln 10)
        self.chain(v, self.re.reciprocal_p::<P>() / V::LN_10)
    }

    #[inline(always)]
    fn log_n<P: Policy, const M: usize>(self) -> Self {
        let v = self.re.log_n_p::<P, M>();
        // d/dx log_M(x) = 1 / (x ln M)
        let ln_m = V::splat(<V::Element as FloatElement>::from_int(M as thermite::LargeInt)).ln_p::<P>();
        self.chain(v, (self.re * ln_m).reciprocal_p::<P>())
    }

    #[inline(always)]
    fn sinc<P: Policy>(self) -> Self {
        let v = self.re.sinc_p::<P>();
        // Only cos is needed for the derivative; cos_p is a dedicated primitive on
        // f32 (cheaper than sin_cos, which would also compute the unused sine).
        let c = self.re.cos_p::<P>();
        // f(x) = sin(x)/x, f'(x) = (cos(x) - sinc(x)) / x, with f'(0) = 0
        let factor = (c - v) / self.re;
        let factor = self.re.is_zero().select(V::ZERO, factor);
        self.chain(v, factor)
    }

    // Default is `sinc(self * PI)`; use the dedicated `sinc_pi`/`cos_pi` primitives.
    // sinc_pi(x) = sin(pi x)/(pi x); f'(x) = (cos(pi x) - sinc_pi(x)) / x, f'(0) = 0.
    #[inline(always)]
    fn sinc_pi<P: Policy>(self) -> Self {
        let v = self.re.sinc_pi_p::<P>();
        let c = self.re.cos_pi_p::<P>();
        let factor = (c - v) / self.re;
        let factor = self.re.is_zero().select(V::ZERO, factor);
        self.chain(v, factor)
    }

    // The default composes `ln(1 - exp(-x))` out of three dual ops, ignoring the
    // dedicated (more accurate) inner `ln1m_expnx`. Use the primitive for the
    // value and the analytic derivative g'(x) = 1/(e^x - 1) (via exp_m1).
    #[inline(always)]
    fn ln1m_expnx<P: Policy>(self) -> Self {
        let v = self.re.ln1m_expnx_p::<P>();
        self.chain(v, self.re.exp_m1_p::<P>().reciprocal_p::<P>())
    }

    #[inline(always)]
    fn ln1m_expnx_ext<P: Policy>(self, lnx: Self) -> Self {
        let v = self.re.ln1m_expnx_ext_p::<P>(lnx.re);
        // g(x) = ln(1 - e^(-x)),  g'(x) = 1 / (e^x - 1).
        // Use exp_m1 (= e^x - 1) instead of exp(x) - 1: same cost, but it avoids
        // catastrophic cancellation near x = 0, where the derivative blows up to 1/x.
        self.chain(v, self.re.exp_m1_p::<P>().reciprocal_p::<P>())
    }
}

impl<V: DualMathVector, const N: usize> SpecializedSpatialMath<Dual<V::Element, N>> for Dual<V, N> {
    #[inline(always)]
    fn l1_norm<P: Policy>(self) -> Self {
        self.abs()
    }

    #[inline(always)]
    fn l2_norm_squared<P: Policy>(self) -> Self {
        self * self
    }

    // The default is `sqrt(self^2)`, which on a dual is a dual square + dual sqrt
    // and yields a 0/0 = NaN derivative at the origin. For a 1-D value the L2 norm
    // is just |x|, whose derivative is sign(x).
    #[inline(always)]
    fn l2_norm<P: Policy>(self) -> Self {
        self.abs()
    }

    // The default routes through `hypot_n_impl`: scaling, dual squaring, a dual
    // sum and a dual sqrt -- expensive, and NaN-derivative at the origin. Compute
    // the primal with the dedicated inner `hypot_n`, then apply the analytic
    // gradient d/dt ||v|| = (sum_k v_k * v_k') / ||v||.
    #[inline(always)]
    fn hypot_n<P: Policy, const K: usize>(values: [Self; K]) -> Self {
        let mut re = [V::ZERO; K];
        let mut k = 0;
        while k < K {
            re[k] = values[k].re;
            k += 1;
        }

        let h = <V as thermite::math::SpatialMathWithPolicy>::hypot_n_p::<P, K>(re);
        let inv = h.reciprocal_p::<P>();

        let mut dual = [V::ZERO; N];
        let mut i = 0;
        while i < N {
            let mut acc = V::ZERO;
            let mut k = 0;
            while k < K {
                acc = re[k].mul_adde(values[k].dual[i], acc);
                k += 1;
            }
            dual[i] = acc * inv;
            i += 1;
        }

        Dual { re: h, dual }
    }

    // Same structure as `hypot_n`, but for 1/||v||. d/dt (1/||v||) = -||v||^-3 * (sum_k v_k v_k').
    #[inline(always)]
    fn inv_hypot_n<P: Policy, const K: usize>(values: [Self; K]) -> Self {
        let mut re = [V::ZERO; K];
        let mut k = 0;
        while k < K {
            re[k] = values[k].re;
            k += 1;
        }

        let ih = <V as thermite::math::SpatialMathWithPolicy>::inv_hypot_n_p::<P, K>(re);
        let factor = (ih * ih * ih).neg(); // -1/||v||^3

        let mut dual = [V::ZERO; N];
        let mut i = 0;
        while i < N {
            let mut acc = V::ZERO;
            let mut k = 0;
            while k < K {
                acc = re[k].mul_adde(values[k].dual[i], acc);
                k += 1;
            }
            dual[i] = factor * acc;
            i += 1;
        }

        Dual { re: ih, dual }
    }
}

impl<V: DualMathVector, const N: usize> SpecializedRealMath<Dual<V::Element, N>> for Dual<V, N> {
    #[inline(always)]
    fn atan2<P: Policy>(self, x: Self) -> Self {
        let v = self.re.atan2_p::<P>(x.re);
        // d/da atan2(a,b) = b/(a^2+b^2); d/db = -a/(a^2+b^2)
        let denom = self.re.mul_adde(self.re, x.re * x.re);
        let inv = denom.reciprocal_p::<P>();
        let mut dual = self.dual;
        let mut i = 0;
        while i < N {
            // (b*a' - a*b') / denom  =  fnma(a, b', b*a') * (1/denom)
            dual[i] = self.re.nmul_adde(x.dual[i], x.re * self.dual[i]) * inv;
            i += 1;
        }
        Dual { re: v, dual }
    }

    // For N >= 3 the default runs a Newton iteration (with a derivative-damping clamp and
    // tolerance branching), so differentiating *through* it would corrupt the dual parts.
    // Instead compute the value with the inner scalar-Newton primitive and apply the
    // implicit-function-theorem derivative: for `t = smoothstep^{-1}(y)`, `dt/dy = 1 / smoothstep'(t)`.
    // `smoothstep_derivative` already carries the `1/(b-a)` edge factor, so its reciprocal is the
    // exact `dt/dy` even with edges (which are treated as constant parameters here).
    #[inline(always)]
    fn inverse_smoothstep<P: Policy, const M: usize>(y: Self, edges: Option<(Self, Self)>) -> Self {
        let edges_re = edges.map(|(a, b)| (a.re, b.re));
        let t = y.re.inverse_smoothstep_p::<P, M>(edges_re);
        let dprime = t.smoothstep_derivative_p::<P, M>(edges_re);
        y.chain(t, dprime.reciprocal_p::<P>())
    }
}
