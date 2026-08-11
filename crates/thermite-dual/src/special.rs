//! Special functions for [`Dual`] via the chain rule (`special` feature).
//!
//! Implements `thermite_special`'s `SpecializedSpecialMath` /
//! `SpecializedRealSpecialMath` for `Dual<V, N>`, so dual vectors gain the full
//! [`SpecialMath`](thermite_special::SpecialMath) /
//! [`RealSpecialMath`](thermite_special::RealSpecialMath) APIs (and `_p`
//! variants).
//!
//! As with the core math module, only the handful of *required* primitives are
//! written by hand (value from the inner primitive, derivative via the chain
//! rule); every default (`erfc`, `logistic_sigmoid`, `softplus`,
//! `hermite`, `chebyshev`, `jacobi`, `legendre`, `gaussian`, `gelu`, `swish`,
//! `algebraic_*`, `gaussian_integral`, ...) composes out of dual arithmetic and
//! is therefore differentiated automatically.
//!
//! ## Not implemented (`todo!()`)
//!
//! `trigamma`, because the Γ-derivative family is not closed under
//! differentiation: ψ₁' is ψ₂, whose derivative is ψ₃, and so on. Adding an
//! order to the trait moves the hole one step out instead of filling it, so the
//! ladder is cut here - one order past what the rest of the family needs.
//! Closing it properly means a general `polygamma(n)`, which *is* closed, since
//! its derivative is `polygamma(n + 1)`.
//!
//! `bessel_j`, because only order 0 exists upstream, so `J_n' = (J_{n-1} -
//! J_{n+1})/2` cannot be formed.
//!
//! Both panic if called; everything that does not depend on them works.

use thermite::math::policy::Policy;
use thermite_special::specialized::{SpecializedRealSpecialMath, SpecializedSpecialMath};
use thermite_special::{RealSpecialMathWithPolicy, SpecialMathWithPolicy};

use thermite::prelude::*;

use crate::Dual;
use crate::math::DualMathVector;

/// Inner vector requirements for the dual special-function library.
pub trait DualSpecialVector: DualMathVector + SpecialMathWithPolicy + RealSpecialMathWithPolicy {}
impl<V> DualSpecialVector for V where V: DualMathVector + SpecialMathWithPolicy + RealSpecialMathWithPolicy {}

// `SpecializedSpecialMath<E>` buys exactly one thing here: `trigamma`, which lives
// only on the specialized trait (see its docs) and is what `digamma`'s derivative
// needs. The element type is spelled as a separate `E` rather than `V::Element`
// because a bound that mentions `V`'s own associated type while computing `V`'s
// bounds is a cycle - the same reason the generic kernels upstream are written
// `V: FloatVector<Element = E> + SpecializedSpecialMath<E>`.
impl<V, E, const N: usize> SpecializedSpecialMath<Dual<E, N>> for Dual<V, N>
where
    V: DualSpecialVector + FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    type ExpIntDetails = Self;

    #[inline(always)]
    fn erf<P: Policy>(self) -> Self {
        let v = self.re.erf_p::<P>();
        // d/dx erf(x) = 2/sqrt(pi) * e^(-x^2)
        let factor = V::FRAC_2_SQRT_PI * (self.re * self.re).neg().exp_p::<P>();
        self.chain(v, factor)
    }

    /// `M` rather than `N` because `N` is already this impl's dual-part count.
    #[inline(always)]
    fn expint<P: Policy, const M: usize>(self) -> Self {
        // Differentiating E_M(x) = \int_1^inf e^-xt / t^M dt under the integral sign
        // pulls down a factor of -t, which is exactly one order lower:
        //   E_M'(x) = -E_{M-1}(x)
        // and at M = 1 that bottoms out in E_0(x) = e^-x / x, plain exp.
        //
        // So the whole thing is a real evaluation plus a chain rule, and the pair comes
        // out of one call because the order recurrence passes through E_{M-1} on its way
        // to E_M. Worth doing beyond the obvious cost saving: the real path guards that
        // recurrence with RECURRENCE_THRESHOLD and swaps in an asymptotic series above
        // it, which the generic dual-arithmetic default this used to inherit does not.
        let (v, prev) = <V as SpecializedSpecialMath<E>>::expint_primal::<P, M>(self.re);

        self.chain(v, -prev)
    }

    #[inline(always)]
    fn lambert_w<P: Policy>(self) -> (Self, Self) {
        let (w0, wm1) = self.re.lambert_w_p::<P>();
        // W'(x) = W / (x (1 + W)) = W / (x*W + x)
        let f0 = w0 / self.re.mul_adde(w0, self.re);
        let fm1 = wm1 / self.re.mul_adde(wm1, self.re);
        (self.chain(w0, f0), self.chain(wm1, fm1))
    }

    #[inline(always)]
    fn tgamma<P: Policy>(self) -> Self {
        let v = self.re.tgamma_p::<P>();
        // Gamma'(x) = Gamma(x) psi(x)
        self.chain(v, v * self.re.digamma_p::<P>())
    }

    #[inline(always)]
    fn lgamma<P: Policy>(self) -> Self {
        let v = self.re.lgamma_p::<P>();
        // d/dx ln|Gamma(x)| = psi(x), on either side of the poles
        self.chain(v, self.re.digamma_p::<P>())
    }

    #[inline(always)]
    fn digamma<P: Policy>(self) -> Self {
        let v = self.re.digamma_p::<P>();
        // psi'(x) = psi_1(x), the trigamma function. Reached through the specialized
        // trait because `trigamma` is deliberately not on the public one.
        self.chain(v, SpecializedSpecialMath::trigamma::<P>(self.re))
    }

    #[inline(always)]
    fn beta<P: Policy>(a: Self, b: Self) -> Self {
        let v = a.re.beta_p::<P>(b.re);

        // B = Gamma(a)Gamma(b)/Gamma(a+b), so ln B = lnGamma(a) + lnGamma(b) - lnGamma(a+b)
        // and dB/da = B (psi(a) - psi(a+b)), dB/db = B (psi(b) - psi(a+b)). The shared
        // psi(a+b) is computed once.
        let psi_ab = (a.re + b.re).digamma_p::<P>();
        let fa = v * (a.re.digamma_p::<P>() - psi_ab);
        let fb = v * (b.re.digamma_p::<P>() - psi_ab);

        // Two independent variables, so both gradients accumulate into one dual part.
        let mut dual = a.dual;
        let mut i = 0;
        while i < N {
            dual[i] = fa.mul_adde(a.dual[i], fb * b.dual[i]);
            i += 1;
        }
        Dual { re: v, dual }
    }

    // psi_1' = psi_2 (tetragamma). Left unimplemented on purpose: the Gamma-derivative
    // family is not closed under differentiation, so every order added to the trait
    // moves this hole one step further out rather than filling it. Closing it for good
    // needs a general `polygamma(n)`, whose derivative is just `polygamma(n + 1)`.
    #[inline(always)]
    fn trigamma<P: Policy>(self) -> Self {
        todo!("Dual trigamma requires the tetragamma function psi_2; see polygamma")
    }

    // TEMP(bessel_j): disabled until orders beyond J_0 exist - see thermite-special/src/lib.rs.
    // Would need adjacent orders J_(n-1), J_(n+1) for J_n' anyway.
    //#[inline(always)]
    //fn bessel_j<P: Policy, const M: usize>(self) -> Self {
    //    todo!()
    //}
}

// Same bound as the `SpecializedSpecialMath` impl above, which this one requires.
impl<V, E, const N: usize> SpecializedRealSpecialMath<Dual<E, N>> for Dual<V, N>
where
    V: DualSpecialVector + FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    #[inline(always)]
    fn erfinv<P: Policy>(self) -> Self {
        let v = self.re.erfinv_p::<P>();
        // d/dx erfinv(x) = (sqrt(pi)/2) e^(erfinv(x)^2)
        let factor = V::FRAC_SQRT_PI_2 * (v * v).exp_p::<P>();
        self.chain(v, factor)
    }

    #[inline(always)]
    fn probit<P: Policy>(self) -> Self {
        let v = self.re.probit_p::<P>();
        // probit = Phi^-1; d/dx = 1/phi(probit(x)) = sqrt(2*pi) e^(v^2/2)
        let factor = V::SQRT_TAU * (v * v * V::HALF).exp_p::<P>();
        self.chain(v, factor)
    }

    #[inline(always)]
    fn lgamma_r<P: Policy>(self) -> (Self, Self) {
        let (v, sign) = self.re.lgamma_r_p::<P>();
        // Same derivative as `lgamma`. The sign is piecewise constant in x, so it
        // carries a zero derivative rather than the incoming one.
        (self.chain(v, self.re.digamma_p::<P>()), Self::constant(sign))
    }
}

/// `Dual` overrides `expint` outright and delegates to the inner vector, so these are
/// never consulted on the hot path - but the real-line defaults are the right answer
/// anyway, since a dual number orders and compares by its real part.
impl<V, E: 'static, const N: usize> thermite_special::specialized::ExpIntDetails<Dual<E, N>, Dual<V, N>>
    for Dual<V, N>
where
    Dual<V, N>: thermite::vector::FloatVector<Element = Dual<E, N>>,
{
}
