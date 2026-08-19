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
//! propagates derivatives with the chain rule `f(a + bε) = f(a) + b*f'(a)ε`.
//! Every *non*-primitive (e.g. `tan`, `sin`, `powi`, `lerp`, `smoothstep`)
//! comes from the trait defaults, which compose out of the dual arithmetic and
//! are therefore differentiated automatically.

use thermite::math::FloatConsts;
use thermite::math::algorithms::reduce_in_place;
use thermite::math::policy::Policy;
use thermite::math::specialized::{
    SpecializedCoreMath, SpecializedRealMath, SpecializedSpatialMath, SpecializedTranscendentalMath,
};
use thermite::math::{PrimalProjection, RealMathWithPolicy};
use thermite::prelude::*;
use thermite::vector::AsFloatVectorWithBitsKernel;

use crate::Dual;
use crate::vector::DualFloatVector;

/// Inner vector requirements for the dual math library: a real float vector that
/// supports the full policy-aware math suite and float constants.
pub trait DualMathVector: DualFloatVector + RealMathWithPolicy + FloatConsts {}
impl<V> DualMathVector for V where V: DualFloatVector + RealMathWithPolicy + FloatConsts {}

impl<V: DualMathVector, const N: usize> PrimalProjection for Dual<V, N> {
    // A constant's derivative parts are identically zero, so tables and cached
    // coefficients live in the inner vector's primal, recursively.
    type Primal = V::Primal;

    #[inline(always)]
    fn from_primal(p: Self::Primal) -> Self {
        Self::constant(V::from_primal(p))
    }

    #[inline(always)]
    fn to_primal(self) -> Self::Primal {
        self.re.to_primal()
    }
}

impl<V: DualMathVector, const N: usize> SpecializedCoreMath<Dual<V::Element, N>> for Dual<V, N> {
    /// The product is a genuine dual multiply (both operands vary), but the addend is
    /// a constant, so only the value part moves. The default would lift `a` into a
    /// `Dual` with `N` zero derivatives and add those too, and `d + 0.0` does not fold
    /// to `d` (it is wrong for `-0.0`), so those adds would survive to run time.
    #[inline(always)]
    fn mul_add_primal<P: Policy>(self, m: Self, a: Self::Primal) -> Self {
        let prod = self * m;

        Dual {
            re: prod.re + V::from_primal(a),
            dual: prod.dual,
        }
    }

    /// As [`mul_add_primal`](SpecializedCoreMath::mul_add_primal), negated. Spelled out
    /// rather than left to the default so the derivative components are negated in place
    /// instead of `self` being negated first and the product rebuilt.
    #[inline(always)]
    fn nmul_add_primal<P: Policy>(self, m: Self, a: Self::Primal) -> Self {
        let prod = self * m;

        // Hand-rolled: `array::map` does not inline inside target_feature code and
        // falls back to scalar.
        let mut dual = prod.dual;
        let mut i = 0;
        while i < N {
            dual[i] = -dual[i];
            i += 1;
        }

        Dual {
            re: V::from_primal(a) - prod.re,
            dual,
        }
    }

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
    // `d/dx tan = 1 + tan^2` is cheaper, since it uses the element type's dedicated
    // `tan` (f32) and avoids the per-component division.
    #[inline(always)]
    fn tan<P: Policy>(self) -> Self {
        let t = self.re.tan_p::<P>();
        self.chain(t, t.mul_adde(t, V::ONE))
    }

    // The default computes `sin_cos(self * PI)`, the generic sine/cosine of x*pi
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
        // 1 / sqrt(1 - x^2) = inverse_sqrt(1 - x^2)
        self.chain(v, self.re.nmul_adde(self.re, V::ONE).inverse_sqrt_p::<P>())
    }

    #[inline(always)]
    fn acos<P: Policy>(self) -> Self {
        let v = self.re.acos_p::<P>();
        // -1 / sqrt(1 - x^2) = -inverse_sqrt(1 - x^2)
        self.chain(v, self.re.nmul_adde(self.re, V::ONE).inverse_sqrt_p::<P>().neg())
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
        // 1 / sqrt(x^2 + 1) = inverse_sqrt(x^2 + 1)
        self.chain(v, self.re.mul_adde(self.re, V::ONE).inverse_sqrt_p::<P>())
    }

    #[inline(always)]
    fn acosh<P: Policy>(self) -> Self {
        let v = self.re.acosh_p::<P>();
        // 1 / sqrt(x^2 - 1) = inverse_sqrt(x^2 - 1)
        self.chain(v, self.re.mul_sube(self.re, V::ONE).inverse_sqrt_p::<P>())
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
    fn exp2_m1<P: Policy>(self) -> Self {
        let v = self.re.exp2_m1_p::<P>();
        // d/dx (2^x - 1) = ln(2) 2^x = ln(2) (v + 1)
        self.chain(v, v.mul_adde(V::LN_2, V::LN_2))
    }

    #[inline(always)]
    fn exp10_m1<P: Policy>(self) -> Self {
        let v = self.re.exp10_m1_p::<P>();
        // d/dx (10^x - 1) = ln(10) 10^x = ln(10) (v + 1)
        self.chain(v, v.mul_adde(V::LN_10, V::LN_10))
    }

    #[inline(always)]
    fn powf<P: Policy>(self, e: Self) -> Self {
        let v = self.re.powf_p::<P>(e.re);
        // d/dx x^y = y x^(y-1) = y * (x^y) / x = e.re * v / x;  d/dy x^y = x^y ln x
        let a = e.re * v / self.re;

        /// Kernel for [`FloatVector::with_bits`]: bitwise-OR all `N` exponent-derivative
        /// components into one accumulator (a [`FloatVectorWithBits`] is a
        /// [`BitwiseVector`], so `|` applies directly to the floats) and report whether
        /// it is all-zero, i.e. whether the exponent is a constant. Returns `None` on
        /// backends without bit access, where the caller falls back to the full path.
        struct ExpIsConstKernel;

        impl<O: FloatVector, const N: usize> AsFloatVectorWithBitsKernel<O, N> for ExpIsConstKernel {
            type Output = bool;

            #[inline(always)]
            fn with_bits<
                W: FloatVectorWithBits<
                        Element = O::Element,
                        Lanes = O::Lanes,
                        Mask = O::Mask,
                        Signed = O::Signed,
                        Unsigned = O::Unsigned,
                        ExtendedPrecision = O::ExtendedPrecision,
                    > + CastVector<O>,
            >(
                self,
                mut v: [W; N],
            ) -> bool {
                reduce_in_place(&mut v, |x, y| x | y);
                v[0].is_all_zero()
            }
        }

        // The d/dy term (x^y ln x) needs a `ln` (a full transcendental) and its
        // x<=0 NaN handling, but is only live when the exponent actually carries a
        // derivative. For the common `x.powf(const)` case every `e.dual` is zero,
        // so detect that and skip the whole d/dy term, falling back to the plain
        // base-direction chain rule. Bit-capable backends do it as a single
        // OR-reduce of the partials' bits, while types without bit access (e.g.
        // Compensated) take the numeric `is_all_zero` per partial instead.
        let exp_is_const = match <V as FloatVector>::with_bits(e.dual, ExpIsConstKernel) {
            Some(is_const) => is_const,
            None => {
                let mut is_const = true;
                let mut i = 0;
                while i < N {
                    is_const &= e.dual[i].is_all_zero();
                    i += 1;
                }
                is_const
            }
        };

        if thermite::likely(exp_is_const) {
            return self.chain(v, a);
        }

        // ln(x) is -inf/NaN for x <= 0; zero the d/dy contribution there so a
        // finite primal (e.g. an integer exponent over a negative base) isn't
        // NaN-poisoned by `b * 0`. Where the exponent varies and x <= 0 the result
        // is already NaN via `v`/`a`, so this is safe.
        let b = (v * self.re.ln_p::<P>()).nz(self.re.cmp_le(V::ZERO));
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
    // division, which is very expensive. Use the inner dedicated `nth_root` for the value
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
    fn log1pmx<P: Policy>(self) -> Self {
        let v = self.re.log1pmx_p::<P>();
        // d/dx [ln(1 + x) - x] = 1/(1 + x) - 1 = -x/(1 + x), taken in that closed form so
        // the derivative never forms the cancelling difference the primal exists to avoid.
        self.chain(v, -self.re / (V::ONE + self.re))
    }

    #[inline(always)]
    fn log2<P: Policy>(self) -> Self {
        let v = self.re.log2_p::<P>();
        // d/dx log2(x) = 1 / (x ln 2) = log2(e) / x
        self.chain(v, self.re.reciprocal_p::<P>() * V::LOG2_E)
    }

    #[inline(always)]
    fn log10<P: Policy>(self) -> Self {
        let v = self.re.log10_p::<P>();
        // d/dx log10(x) = 1 / (x ln 10) = log10(e) / x
        self.chain(v, self.re.reciprocal_p::<P>() * V::LOG10_E)
    }

    #[inline(always)]
    fn log2_p1<P: Policy>(self) -> Self {
        let v = self.re.log2_p1_p::<P>();
        // d/dx log2(1 + x) = 1 / ((1 + x) ln 2) = log2(e) / (1 + x)
        self.chain(v, (V::ONE + self.re).reciprocal_p::<P>() * V::LOG2_E)
    }

    #[inline(always)]
    fn log10_p1<P: Policy>(self) -> Self {
        let v = self.re.log10_p1_p::<P>();
        // d/dx log10(1 + x) = 1 / ((1 + x) ln 10) = log10(e) / (1 + x)
        self.chain(v, (V::ONE + self.re).reciprocal_p::<P>() * V::LOG10_E)
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
        // Only cos is needed for the derivative, and cos_p is a dedicated primitive on
        // f32 (cheaper than sin_cos, which would also compute the unused sine).
        let c = self.re.cos_p::<P>();
        // f(x) = sin(x)/x, f'(x) = (cos(x) - sinc(x)) / x, with f'(0) = 0
        let factor = (c - v) / self.re;
        let factor = self.re.is_zero().select(V::ZERO, factor);
        self.chain(v, factor)
    }

    /// The trait default guards `x == 0` with a select, which is right for the value and
    /// wrong for the gradient: `x ln y` is *linear* in `x`, so `d/dx` is `ln y` at the origin
    /// like everywhere else, not the zero a select would propagate. That gradient is exactly
    /// what a cross-entropy needs at a probability that has reached zero, so it is worth the
    /// override rather than inheriting a silent zero.
    #[inline(always)]
    fn xlogy<P: Policy>(self, y: Self) -> Self {
        let v = self.re.xlogy_p::<P>(y.re);
        // d/dx = ln y, d/dy = x/y.
        let ln_y = y.re.ln_p::<P>();
        let x_over_y = self.re / y.re;

        let mut dual = self.dual;
        let mut i = 0;
        while i < N {
            dual[i] = x_over_y.mul_adde(y.dual[i], ln_y * self.dual[i]);
            i += 1;
        }

        Dual { re: v, dual }
    }

    /// [`xlogy`](Self::xlogy)'s reasoning, one argument shifted: `d/dx` is `ln(1+y)` and
    /// `d/dy` is `x/(1+y)`.
    #[inline(always)]
    fn xlog1py<P: Policy>(self, y: Self) -> Self {
        let v = self.re.xlog1py_p::<P>(y.re);
        let ln_y = y.re.ln_1p_p::<P>();
        let x_over_y = self.re / (V::ONE + y.re);

        let mut dual = self.dual;
        let mut i = 0;
        while i < N {
            dual[i] = x_over_y.mul_adde(y.dual[i], ln_y * self.dual[i]);
            i += 1;
        }

        Dual { re: v, dual }
    }

    #[inline(always)]
    fn atanhc<P: Policy>(self) -> Self {
        let v = self.re.atanhc_p::<P>();
        // f(x) = atanh(x)/x, f'(x) = (1/(1-x^2) - atanhc(x)) / x, with f'(0) = 0 since f is
        // even. Same shape and same guard as `sinc` above: the quotient is 0/0 at the origin.
        //
        // Known limitation, shared with `sinc`'s override and inherited from the same shape:
        // both terms tend to 1 and differ by 2x^2/3, so the subtraction loses about
        // 1.5*eps/x^2 in relative terms, a few ulp by |x| = 0.2, but ~2e-4 at 1e-6. The
        // primal is unaffected, and only the derivative cancels. Fixing it properly wants an
        // `atanhc_m1` primitive (atanh(x)/x - 1) so that (g-1) - (f-1) is formed from two
        // small quantities instead. `log1pmx` alone does not get there, since
        // atanh(x) - x = (log1pmx(x) - log1pmx(-x))/2 cancels its own x^2/2 terms.
        let d = (V::ONE - self.re.square()).reciprocal_p::<P>();
        let factor = (d - v) / self.re;
        let factor = self.re.is_zero().select(V::ZERO, factor);
        self.chain(v, factor)
    }

    #[inline(always)]
    fn sinhc<P: Policy>(self) -> Self {
        let v = self.re.sinhc_p::<P>();
        let c = self.re.cosh_p::<P>();
        // f(x) = sinh(x)/x, f'(x) = (cosh(x) - sinhc(x)) / x, with f'(0) = 0. Same shape as
        // `sinc` above, and the same reason for the guard: the quotient is 0/0 at the origin.
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
    // sum and a dual sqrt, which is expensive and NaN-derivative at the origin. Compute
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
        // d||v|| is undefined at the origin: h == 0 -> 1/h = inf, dotted with the
        // zero numerator -> NaN. Pin the gradient to 0 there instead.
        let inv = h.reciprocal_p::<P>().nz(h.is_zero());

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
        // -1/||v||^3, undefined at the origin (ih = inf there, and the cube can
        // overflow near it); zero the gradient wherever the factor isn't finite so
        // it can't poison `factor * acc` into a NaN.
        let factor = (ih * ih * ih).neg();
        let factor = factor.zz(factor.is_finite());

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
