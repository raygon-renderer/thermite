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
//! rule); every default (`erfc`, `expint`, `logistic_sigmoid`, `softplus`,
//! `hermite`, `chebyshev`, `jacobi`, `legendre`, `gaussian`, `gelu`, `swish`,
//! `algebraic_*`, `gaussian_integral`, ...) composes out of dual arithmetic and
//! is therefore differentiated automatically.
//!
//! ## Not implemented (`todo!()`)
//!
//! The Γ-family derivatives (`tgamma`, `lgamma`, `beta`, `lgamma_r`) all require
//! the digamma function ψ, which `thermite_special` does not provide. `bessel_j`
//! only implements order 0 upstream, so its derivative `J_n' = (J_{n-1} -
//! J_{n+1})/2` cannot be formed. These panic if called; everything that does not
//! depend on them works.

use thermite::math::policy::Policy;
use thermite_special::specialized::{SpecializedRealSpecialMath, SpecializedSpecialMath};
use thermite_special::{RealSpecialMathWithPolicy, SpecialMathWithPolicy};

use crate::Dual;
use crate::math::DualMathVector;

/// Inner vector requirements for the dual special-function library.
pub trait DualSpecialVector: DualMathVector + SpecialMathWithPolicy + RealSpecialMathWithPolicy {}
impl<V> DualSpecialVector for V where V: DualMathVector + SpecialMathWithPolicy + RealSpecialMathWithPolicy {}

impl<V: DualSpecialVector, const N: usize> SpecializedSpecialMath<Dual<V::Element, N>> for Dual<V, N> {
    #[inline(always)]
    fn erf<P: Policy>(self) -> Self {
        let v = self.re.erf_p::<P>();
        // d/dx erf(x) = 2/sqrt(pi) * e^(-x^2)
        let factor = V::FRAC_2_SQRT_PI * (self.re * self.re).neg().exp_p::<P>();
        self.chain(v, factor)
    }

    #[inline(always)]
    fn lambert_w<P: Policy>(self) -> (Self, Self) {
        let (w0, wm1) = self.re.lambert_w_p::<P>();
        // W'(x) = W / (x (1 + W)) = W / (x*W + x)
        let f0 = w0 / self.re.mul_adde(w0, self.re);
        let fm1 = wm1 / self.re.mul_adde(wm1, self.re);
        (self.chain(w0, f0), self.chain(wm1, fm1))
    }

    // --- require the digamma function psi, which thermite-special lacks ---
    #[inline(always)]
    fn tgamma<P: Policy>(self) -> Self {
        todo!("Dual tgamma requires the digamma function (not provided by thermite-special)")
    }

    #[inline(always)]
    fn lgamma<P: Policy>(self) -> Self {
        todo!("Dual lgamma requires the digamma function (not provided by thermite-special)")
    }

    #[inline(always)]
    fn beta<P: Policy>(_a: Self, _b: Self) -> Self {
        todo!("Dual beta requires the digamma function (not provided by thermite-special)")
    }

    // --- requires adjacent Bessel orders; thermite-special only implements J_0 ---
    #[inline(always)]
    fn bessel_j<P: Policy, const M: usize>(self) -> Self {
        todo!("Dual bessel_j requires adjacent orders J_(n-1), J_(n+1); only J_0 is available")
    }
}

impl<V: DualSpecialVector, const N: usize> SpecializedRealSpecialMath<Dual<V::Element, N>> for Dual<V, N> {
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

    // requires the digamma function psi (for ln|Gamma|'); sign is locally constant
    #[inline(always)]
    fn lgamma_r<P: Policy>(self) -> (Self, Self) {
        todo!("Dual lgamma_r requires the digamma function (not provided by thermite-special)")
    }
}
