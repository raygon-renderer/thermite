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
//! The Gamma-derivative family is closed under differentiation through
//! `polygamma`: its runtime order means `psi_n' = psi_{n+1}` is just `n + 1`, so
//! `digamma`, `trigamma` and `polygamma` itself all differentiate to any nesting
//! depth. (`trigamma` was a `todo!()` until polygamma existed, because each fixed
//! order's derivative needed the next order.)
//!
//! The Bessel families differentiate through identities that reach **down** one
//! order, never up (see the note above `bessel_i` below, and the entry it
//! corrects). This doc used to say `bessel_j` was unimplemented "because only
//! order 0 exists upstream, so `J_n' = (J_{n-1} - J_{n+1})/2` cannot be formed".
//! Both halves of that were wrong. It is implemented.
//!
//! Airy is the easiest of the lot, because `w'' = x w` **is** its definition: the
//! derivative of each value is the other one, and the derivative of that is `x`
//! times the first. The scaled forms carry the scaling's own derivative,
//! `d\zeta/dx = \sqrt{x}`, which does not cancel.
//!
//! ## Not implemented (`todo!()`)
//!
//! The runtime-order Bessel entries `bessel_iv` / `bessel_kv` / `bessel_jv` /
//! `bessel_yv`, which inherit the composite defaults. Nothing else here panics.

use thermite::math::PrimalProjection;
use thermite::math::policy::Policy;
use thermite_special::specialized::{
    ShTable, SpecializedRealSpecialMath, SpecializedSpecialMath, sh_eval_d_impl, sh_eval_mixed_impl,
};
use thermite_special::elliptic::EllipticConsts;
use thermite_special::{RealSpecialMathWithPolicy, SpecialMathWithPolicy};

use thermite::prelude::*;

use crate::Dual;
use crate::math::DualMathVector;

/// Inner vector requirements for the dual special-function library.
pub trait DualSpecialVector: DualMathVector + SpecialMathWithPolicy + RealSpecialMathWithPolicy {}
impl<V> DualSpecialVector for V where V: DualMathVector + SpecialMathWithPolicy + RealSpecialMathWithPolicy {}

/// The elliptic kernels (`carlson`, `ellint`) are generic over any float vector whose element
/// carries this one constant, so lifting it is all a dual element needs for the whole family
/// to run in dual arithmetic: the Carlson duplication and the AGM are contractive algebraic
/// iterations, and the chain rule through them is the derivative. The threshold itself is a
/// constant of the computation, so its dual part is zero.
impl<E: crate::DualValue + thermite::element::FloatElement + EllipticConsts, const N: usize> EllipticConsts for Dual<E, N> {
    const CARLSON_THRESH: Self = Dual::constant(E::CARLSON_THRESH);
    const RC_SERIES_THRESH: Self = Dual::constant(E::RC_SERIES_THRESH);
}

// `SpecializedSpecialMath<E>` buys exactly one thing here: `trigamma`, which lives
// only on the specialized trait (see its docs) and is what `digamma`'s derivative
// needs. The element type is spelled as a separate `E` rather than `V::Element`
// because a bound that mentions `V`'s own associated type while computing `V`'s
// bounds is a cycle, the same reason the generic kernels upstream are written
// `V: FloatVector<Element = E> + SpecializedSpecialMath<E>`.
impl<V, E, const N: usize> SpecializedSpecialMath<Dual<E, N>> for Dual<V, N>
where
    V: DualSpecialVector + FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
    type ExpIntDetails = Self;
    const LAGUERRE_PRODUCT_SEED_CAP: i32 = V::LAGUERRE_PRODUCT_SEED_CAP;

    #[inline(always)]
    fn erf<P: Policy>(self) -> Self {
        let v = self.re.erf_p::<P>();
        // d/dx erf(x) = 2/sqrt(pi) * e^(-x^2)
        let factor = V::FRAC_2_SQRT_PI * (self.re * self.re).neg().exp_p::<P>();
        self.chain(v, factor)
    }

    /// `M` rather than `N` because `N` is already this impl's dual-part count.
    #[inline(always)]
    fn expint_n<P: Policy, const M: usize>(self) -> Self {
        // Differentiating E_M(x) = \int_1^inf e^-xt / t^M dt under the integral sign
        // pulls down a factor of -t, which is exactly one order lower:
        //   E_M'(x) = -E_{M-1}(x)
        // and at M = 1 that bottoms out in E_0(x) = e^-x / x, plain exp.
        //
        // So the whole thing is a real evaluation plus a chain rule, and the pair comes
        // out of one call because the order recurrence passes through E_{M-1} on its way
        // to E_M. Worth doing beyond the obvious cost saving: the real path guards that
        // recurrence with RECURRENCE_THRESHOLD and swaps in an asymptotic series above
        // it, which the generic dual-arithmetic default does not.
        let (v, prev) = <V as SpecializedSpecialMath<E>>::expint_primal_n::<P, M>(self.re);

        self.chain(v, -prev)
    }

    #[inline(always)]
    fn expint<P: Policy>(self, n: u32) -> Self {
        let (v, prev) = <V as SpecializedSpecialMath<E>>::expint_primal::<P>(self.re, n);
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

    #[inline(always)]
    fn trigamma<P: Policy>(self) -> Self {
        // Reached through the specialized trait, like `digamma`'s derivative above.
        let v = SpecializedSpecialMath::trigamma::<P>(self.re);
        // psi_1' = psi_2, reachable now that `polygamma`'s runtime order closed the
        // Gamma-derivative family (this was a `todo!()` while orders were fixed).
        self.chain(v, self.re.polygamma_p::<P>(2))
    }

    /// `zeta` and `zetac` differentiate from a _second kernel_, not a chain rule over the
    /// first: `zeta'(s) = -sum ln(n) n^-s` has no expression in terms of `zeta` itself, unlike
    /// the way `polygamma` closes on its own family. The inner vector computes both together,
    /// which is far cheaper than twice, since they share every transcendental.
    ///
    /// The two functions differ by a constant, so one derivative serves both.
    #[inline(always)]
    fn zeta<P: Policy>(self) -> Self {
        let (v, d) = <V as SpecializedSpecialMath<E>>::zeta_with_deriv::<P, false>(self.re);
        self.chain(v, d)
    }

    #[inline(always)]
    fn zetac<P: Policy>(self) -> Self {
        let (v, d) = <V as SpecializedSpecialMath<E>>::zeta_with_deriv::<P, true>(self.re);
        self.chain(v, d)
    }

    // --- Bessel ---
    //
    // All four families close analytically under differentiation _within_ the orders already
    // available, because every identity here reaches DOWN one order:
    //
    //     I_N' =  I_{N-1} - (N/x) I_N      K_N' = -K_{N-1} - (N/x) K_N
    //     J_N' =  J_{N-1} - (N/x) J_N      Y_N' =  Y_{N-1} - (N/x) Y_N
    //
    // and at N = 0 the negative order folds back (`I_{-1} = I_1`, `J_{-1} = -J_1`), so the
    // pattern never needs order N+1. That matters more than it looks: the textbook spelling
    // `J_N' = (J_{N-1} - J_{N+1})/2` DOES need N+1, and believing it was the reason
    // `bessel_j` was disabled crate-wide with a note saying orders beyond J_0 had to exist
    // first. They did not.
    //
    // `M` rather than `N` throughout: `N` is already this impl's dual-part count.
    //
    // Each `*_with_deriv` computes both halves in one pass (the recurrences produce order
    // N-1 alongside N for free), so this is one evaluation, not two.

    #[inline(always)]
    fn bessel_i<P: Policy, const M: i32>(self) -> Self {
        let (v, d) = <V as SpecializedSpecialMath<E>>::bessel_i_with_deriv::<P, M, false>(self.re);
        self.chain(v, d)
    }

    #[inline(always)]
    fn bessel_i_scaled<P: Policy, const M: i32>(self) -> Self {
        let (v, d) = <V as SpecializedSpecialMath<E>>::bessel_i_with_deriv::<P, M, true>(self.re);
        self.chain(v, d)
    }

    #[inline(always)]
    fn bessel_k<P: Policy, const M: i32>(self) -> Self {
        let (v, d) = <V as SpecializedSpecialMath<E>>::bessel_k_with_deriv::<P, M, false>(self.re);
        self.chain(v, d)
    }

    #[inline(always)]
    fn bessel_k_scaled<P: Policy, const M: i32>(self) -> Self {
        let (v, d) = <V as SpecializedSpecialMath<E>>::bessel_k_with_deriv::<P, M, true>(self.re);
        self.chain(v, d)
    }

    #[inline(always)]
    fn bessel_j<P: Policy, const M: i32>(self) -> Self {
        let (v, d) = <V as SpecializedSpecialMath<E>>::bessel_j_with_deriv::<P, M>(self.re);
        self.chain(v, d)
    }

    #[inline(always)]
    fn bessel_y<P: Policy, const M: i32>(self) -> Self {
        let (v, d) = <V as SpecializedSpecialMath<E>>::bessel_y_with_deriv::<P, M>(self.re);
        self.chain(v, d)
    }


    // ---- spherical Bessel ------------------------------------------------------------------
    //
    // Same shape as the cylindrical families above: the derivative identity reaches DOWN one
    // order, `f_n' = f_{n-1} - ((n+1)/x) f_n`, and the recurrence passes through `n-1` on its
    // way, so `*_with_deriv` costs no more than the value alone.
    //
    // The `n+1` is not a typo for the cylindrical `n`: differentiating the `sqrt(pi/2x)` that
    // relates the two normalizations contributes the extra half.

    #[inline(always)]
    fn sph_bessel_j_n<P: Policy, const M: usize>(self) -> Self {
        let (v, d) = <V as SpecializedSpecialMath<E>>::sph_bessel_j_with_deriv_n::<P, M>(self.re);
        self.chain(v, d)
    }

    #[inline(always)]
    fn sph_bessel_y_n<P: Policy, const M: usize>(self) -> Self {
        let (v, d) = <V as SpecializedSpecialMath<E>>::sph_bessel_y_with_deriv_n::<P, M>(self.re);
        self.chain(v, d)
    }

    #[inline(always)]
    fn sph_bessel_i_n<P: Policy, const M: usize>(self) -> Self {
        let (v, d) = <V as SpecializedSpecialMath<E>>::sph_bessel_i_with_deriv_n::<P, M, false>(self.re);
        self.chain(v, d)
    }

    #[inline(always)]
    fn sph_bessel_i_scaled_n<P: Policy, const M: usize>(self) -> Self {
        let (v, d) = <V as SpecializedSpecialMath<E>>::sph_bessel_i_with_deriv_n::<P, M, true>(self.re);
        self.chain(v, d)
    }

    #[inline(always)]
    fn sph_bessel_k_n<P: Policy, const M: usize>(self) -> Self {
        let (v, d) = <V as SpecializedSpecialMath<E>>::sph_bessel_k_with_deriv_n::<P, M, false>(self.re);
        self.chain(v, d)
    }

    #[inline(always)]
    fn sph_bessel_k_scaled_n<P: Policy, const M: usize>(self) -> Self {
        let (v, d) = <V as SpecializedSpecialMath<E>>::sph_bessel_k_with_deriv_n::<P, M, true>(self.re);
        self.chain(v, d)
    }

    // The runtime-order twins: the same pairs, from the runtime-order kernels.

    #[inline(always)]
    fn sph_bessel_j<P: Policy>(self, n: u32) -> Self {
        let (v, d) = <V as SpecializedSpecialMath<E>>::sph_bessel_j_with_deriv::<P>(self.re, n);
        self.chain(v, d)
    }

    #[inline(always)]
    fn sph_bessel_y<P: Policy>(self, n: u32) -> Self {
        let (v, d) = <V as SpecializedSpecialMath<E>>::sph_bessel_y_with_deriv::<P>(self.re, n);
        self.chain(v, d)
    }

    #[inline(always)]
    fn sph_bessel_i<P: Policy>(self, n: u32) -> Self {
        let (v, d) = <V as SpecializedSpecialMath<E>>::sph_bessel_i_with_deriv::<P, false>(self.re, n);
        self.chain(v, d)
    }

    #[inline(always)]
    fn sph_bessel_i_scaled<P: Policy>(self, n: u32) -> Self {
        let (v, d) = <V as SpecializedSpecialMath<E>>::sph_bessel_i_with_deriv::<P, true>(self.re, n);
        self.chain(v, d)
    }

    #[inline(always)]
    fn sph_bessel_k<P: Policy>(self, n: u32) -> Self {
        let (v, d) = <V as SpecializedSpecialMath<E>>::sph_bessel_k_with_deriv::<P, false>(self.re, n);
        self.chain(v, d)
    }

    #[inline(always)]
    fn sph_bessel_k_scaled<P: Policy>(self, n: u32) -> Self {
        let (v, d) = <V as SpecializedSpecialMath<E>>::sph_bessel_k_with_deriv::<P, true>(self.re, n);
        self.chain(v, d)
    }
    // ---- Airy -----------------------------------------------------------------------------
    //
    // Airy is the easiest family in the crate to differentiate, because it is _defined_ by
    // `w'' = x w`. So the derivative of a value is the other value, and the derivative of that
    // is `x` times the first:
    //
    //     Ai' = Ai'          Ai'' = x Ai
    //     Bi' = Bi'          Bi'' = x Bi
    //
    // Nothing differentiates the Bessel machinery underneath, and every order needed is one
    // the kernels already produce. Contrast the Bessel families above, where the derivative
    // identities had to be spelled so they reach DOWN one order rather than up.
    //
    // The single-value entries take two inner calls rather than one `airy`, deliberately: `Ai`
    // and `Ai'` come from different Bessel orders, so a Dual `airy_ai` needs both passes no
    // matter what, but it needs neither of `Bi`'s, and `airy_ai` skips `I` inside the passes
    // it does run. Calling the tuple would compute two values and one continued fraction that
    // are then dropped. `airy` itself is one call, because there it is all four.

    #[inline(always)]
    fn airy_tuple<P: Policy>(self) -> (Self, Self, Self, Self) {
        let (ai, aip, bi, bip) = <V as SpecializedSpecialMath<E>>::airy_tuple::<P>(self.re);
        let x = self.re;

        (
            self.chain(ai, aip),
            self.chain(aip, x * ai),
            self.chain(bi, bip),
            self.chain(bip, x * bi),
        )
    }

    #[inline(always)]
    fn airy_ai<P: Policy>(self) -> Self {
        let v = <V as SpecializedSpecialMath<E>>::airy_ai::<P>(self.re);
        let d = <V as SpecializedSpecialMath<E>>::airy_ai_prime::<P>(self.re);
        self.chain(v, d)
    }

    #[inline(always)]
    fn airy_ai_prime<P: Policy>(self) -> Self {
        let v = <V as SpecializedSpecialMath<E>>::airy_ai_prime::<P>(self.re);
        let ai = <V as SpecializedSpecialMath<E>>::airy_ai::<P>(self.re);
        self.chain(v, self.re * ai)
    }

    #[inline(always)]
    fn airy_bi<P: Policy>(self) -> Self {
        let v = <V as SpecializedSpecialMath<E>>::airy_bi::<P>(self.re);
        let d = <V as SpecializedSpecialMath<E>>::airy_bi_prime::<P>(self.re);
        self.chain(v, d)
    }

    #[inline(always)]
    fn airy_bi_prime<P: Policy>(self) -> Self {
        let v = <V as SpecializedSpecialMath<E>>::airy_bi_prime::<P>(self.re);
        let bi = <V as SpecializedSpecialMath<E>>::airy_bi::<P>(self.re);
        self.chain(v, self.re * bi)
    }

    // The scaled forms carry the scaling's own derivative, which does not cancel.
    //
    // With `zeta = (2/3) x^{3/2}` and `s = dzeta/dx = sqrt(x)` on the positive axis, the
    // product rule gives, writing `A = e^zeta Ai` and so on:
    //
    //     A'  =  s A + P           P' =  s P + x A
    //     B'  = -s B + Q           Q' = -s Q + x B
    //
    // where `P = e^zeta Ai'` and `Q = e^-zeta Bi'`. Below the origin nothing is scaled, and
    // `s = sqrt(max(x, 0))` is then zero, which collapses every line above to the unscaled
    // identity. That is why `s` is written with a `max` rather than a select: the branchless
    // form is also the one that is correct on both sides.

    #[inline(always)]
    fn airy_tuple_scaled<P: Policy>(self) -> (Self, Self, Self, Self) {
        let (ai, aip, bi, bip) = <V as SpecializedSpecialMath<E>>::airy_tuple_scaled::<P>(self.re);
        let x = self.re;
        let s = x.max(V::ZERO).sqrt();

        (
            self.chain(ai, s.mul_adde(ai, aip)),
            self.chain(aip, s.mul_adde(aip, x * ai)),
            self.chain(bi, bip - s * bi),
            self.chain(bip, x.mul_sube(bi, s * bip)),
        )
    }

    #[inline(always)]
    fn airy_ai_scaled<P: Policy>(self) -> Self {
        let v = <V as SpecializedSpecialMath<E>>::airy_ai_scaled::<P>(self.re);
        let p = <V as SpecializedSpecialMath<E>>::airy_ai_prime_scaled::<P>(self.re);
        let s = self.re.max(V::ZERO).sqrt();
        self.chain(v, s.mul_adde(v, p))
    }

    #[inline(always)]
    fn airy_ai_prime_scaled<P: Policy>(self) -> Self {
        let v = <V as SpecializedSpecialMath<E>>::airy_ai_prime_scaled::<P>(self.re);
        let a = <V as SpecializedSpecialMath<E>>::airy_ai_scaled::<P>(self.re);
        let s = self.re.max(V::ZERO).sqrt();
        self.chain(v, s.mul_adde(v, self.re * a))
    }

    #[inline(always)]
    fn airy_bi_scaled<P: Policy>(self) -> Self {
        let v = <V as SpecializedSpecialMath<E>>::airy_bi_scaled::<P>(self.re);
        let q = <V as SpecializedSpecialMath<E>>::airy_bi_prime_scaled::<P>(self.re);
        let s = self.re.max(V::ZERO).sqrt();
        self.chain(v, q - s * v)
    }

    #[inline(always)]
    fn airy_bi_prime_scaled<P: Policy>(self) -> Self {
        let v = <V as SpecializedSpecialMath<E>>::airy_bi_prime_scaled::<P>(self.re);
        let b = <V as SpecializedSpecialMath<E>>::airy_bi_scaled::<P>(self.re);
        let s = self.re.max(V::ZERO).sqrt();
        self.chain(v, self.re.mul_sube(b, s * v))
    }

    #[inline(always)]
    fn polygamma<P: Policy>(self, n: u32) -> Self {
        let v = self.re.polygamma_p::<P>(n);
        // The whole reason polygamma's order is a runtime scalar: psi_n' = psi_{n+1},
        // so the chain rule needs only n + 1 and the family closes at every depth.
        self.chain(v, self.re.polygamma_p::<P>(n + 1))
    }

    /// `Li_s'(z) = Li_{s-1}(z) / z`: the order steps down by one, in whichever class it
    /// was given, so the family closes at every depth the same way `polygamma` does. At the
    /// origin the ratio's limit is 1 (`Li_{s-1}(z) ~ z` there).
    ///
    /// The `Real` payload is this vector's element, a dual number. Only its value is used.
    /// A derivative with respect to the _order_ (`-sum ln k z^k / k^s`, its own Dirichlet
    /// series, like `zeta'`) is not implemented, so a payload carrying a non-zero dual part
    /// is refused loudly rather than silently dropped.
    #[inline(always)]
    fn polylog<P: Policy>(
        self,
        order: thermite_special::PolylogOrder<Dual<E, N>, <<Self as GenericVector>::Signed as GenericVector>::Element>,
    ) -> Self {
        use thermite_special::PolylogOrder;
        let (order, lower) = match order {
            PolylogOrder::Integer(n) => (PolylogOrder::Integer(n), PolylogOrder::Integer(n - <_ as Element>::ONE)),
            PolylogOrder::Real(s) => {
                if s.dual.iter().any(|d| *d != E::ZERO) {
                    todo!("polylog's derivative with respect to the order is not implemented; pass a constant order")
                }
                (PolylogOrder::Real(s.re), PolylogOrder::Real(s.re - E::ONE))
            }
        };
        let v = self.re.polylog_p::<P>(order);
        let ratio = self.re.polylog_p::<P>(lower) / self.re;
        let factor = self.re.cmp_eq(V::ZERO).select(V::ONE, ratio);
        self.chain(v, factor)
    }

    // TEMP(bessel_j): disabled until orders beyond J_0 exist. See thermite-special/src/lib.rs.
    // Would need adjacent orders J_(n-1), J_(n+1) for J_n' anyway.
    //#[inline(always)]
    //fn bessel_j<P: Policy, const M: i32>(self) -> Self {
    //    todo!()
    //}
}

// --- Spherical harmonics: seeding-aware fast paths ---
//
// The generic default would build a coefficient table IN DUAL ARITHMETIC (taking
// square roots of dual numbers whose derivatives are identically zero) and then run
// the whole recurrence with a derivative riding along every operation. Both are
// avoidable whenever the inputs are seeded the way callers actually seed them, and the
// two cases worth catching are cheap to recognise at runtime because a seeded dual part
// is a splat of exactly 0 or exactly 1.
//
// Nothing here narrows the impl's bounds. The gradients come from the free function
// `sh_eval_d_impl`, which needs only `FloatVector`, rather than from the `_d` trait
// method, which lives on `RealPrimalMath` and is deliberately absent on `Dual`. So
// nested `Dual<Dual<..>>` keeps working and simply recurses into its own fast paths.

/// What [`classify`] found in one input's dual part.
///
/// Three counters rather than an enum, so the scan that fills them is straight-line
/// arithmetic: no early exit, no `Option` state machine, nothing that stops LLVM from
/// unrolling a loop whose trip count is a const generic.
#[derive(Clone, Copy)]
struct Seeding {
    /// Components that are neither exactly zero nor exactly one, across all lanes.
    other: u32,
    /// Components that are exactly one across all lanes.
    ones: u32,
    /// Sum of the indices of those components, i.e. the slot itself once `ones == 1`.
    slot: u32,
}

impl Seeding {
    /// The input does not vary: every component is zero.
    #[inline(always)]
    fn is_constant(self) -> bool {
        (self.other | self.ones) == 0
    }

    /// The input is a basis vector: one component is one and the rest are zero.
    #[inline(always)]
    fn is_unit(self) -> bool {
        // `&`, not `&&`. Both halves are already computed, and short-circuiting one
        // integer comparison buys a branch rather than saving work.
        (self.other == 0) & (self.ones == 1)
    }
}

/// Classifies a dual part, branchlessly.
///
/// Each comparison is over all lanes, so a partially seeded register (some lanes a
/// variable, some not) correctly lands in `other` and takes the general path.
#[inline(always)]
fn classify<V: FloatVector, const N: usize>(dual: &[V; N]) -> Seeding {
    let mut other = 0;
    let mut ones = 0;
    let mut slot = 0;

    let mut i = 0;
    while i < N {
        let is_zero = dual[i].cmp_eq(V::ZERO).all() as u32;
        let is_one = dual[i].cmp_eq(V::ONE).all() as u32;

        // A component cannot be both, so the two flags are disjoint and `1 - (a | b)`
        // is exactly "neither".
        other += 1 - (is_zero | is_one);
        ones += is_one;
        slot += is_one * i as u32;

        i += 1;
    }

    Seeding { other, ones, slot }
}

// Same bound as the `SpecializedSpecialMath` impl above, which this one requires, plus the
// real-only specialized trait for the `_with_deriv` forms the overrides below read.
impl<V, E, const N: usize> SpecializedRealSpecialMath<Dual<E, N>> for Dual<V, N>
where
    V: DualSpecialVector + FloatVector<Element = E> + SpecializedSpecialMath<E> + SpecializedRealSpecialMath<E>,
{
    /// The easiest rule in the file: `C' = cos(pi x^2/2)` and `S' = sin(pi x^2/2)` are the
    /// integrands the pair is defined by.
    ///
    /// It reuses the kernel's own two-word phase rather than `x*x*0.5`, and has to.
    /// Past the crossover the _values_ have settled to `1/2` plus a `1/(pi x)` ripple,
    /// but the derivatives are still oscillating at full amplitude, so a phase that is
    /// merely good enough for the values is nowhere near good enough here: at
    /// `x = 98765` the naive phase is 5.3e-6 off, which is the whole derivative.
    #[inline(always)]
    fn fresnel<P: Policy>(self) -> (Self, Self) {
        let (c, s) = self.re.fresnel_p::<P>();
        let (sin_t, cos_t) = thermite_special::specialized::fresnel_phase::<P, E, V>(self.re.abs())
            .sincos_pi_p::<P>();
        (self.chain(c, cos_t), self.chain(s, sin_t))
    }

    /// `Si' = sin(x)/x` and `Ci' = cos(x)/x`.
    ///
    /// Both are correct on the negative axis without a sign fold, which is worth a
    /// line because the two functions get there differently. `Si` is odd and `sinc` is
    /// even, so `Si'(-x) = Si'(x)` already. `Ci` is evaluated at `|x|`, so the chain
    /// rule wants `sign(x) cos(|x|)/|x|`, and `cos` being even that is exactly
    /// `cos(x)/x`.
    #[inline(always)]
    fn sici<P: Policy>(self) -> (Self, Self) {
        // Si' = sin x / x, Ci' = cos x / x (on the negative axis too, since the kernel takes
        // Ci at |x|): one sin_cos and one reciprocal serve both. Si' is exactly 1 at the
        // origin, where the quotient is 0/0.
        let (si, ci) = self.re.sici_p::<P>();
        let (sin_x, cos_x) = self.re.sin_cos_p::<P>();
        let inv_x = self.re.approx_reciprocal_p::<P>();
        // TODO: Revisit this with better precision?
        let dsi = self.re.is_zero().select(V::ONE, sin_x * inv_x);
        (self.chain(si, dsi), self.chain(ci, cos_x * inv_x))
    }

    /// Delegates to the inner vector: the table is primal-typed at every layer, so
    /// `V` fills the exact table this layer needs, and a real inner vector splats
    /// its _compile-time_ constants instead of computing closed forms. A nested dual
    /// recurses into this same shortcut.
    #[inline(always)]
    fn spherical_harmonics_table<P: Policy, const L: usize, const M: usize, const CS: bool>(
        table: &mut ShTable<<V as PrimalProjection>::Primal, M>,
    ) {
        V::spherical_harmonics_table_p::<P, L, M, CS>(table);
    }

    /// Evaluates from a prebuilt primal table, with the same seeding shortcuts as
    /// [`spherical_harmonics`](Self::spherical_harmonics), but reusing the caller's
    /// cached table instead of rebuilding one.
    #[inline(always)]
    fn spherical_harmonics_with<P: Policy, const L: usize, const M: usize>(
        table: &ShTable<<V as PrimalProjection>::Primal, M>,
        x: Self,
        y: Self,
        z: Self,
        out: &mut [Self; M],
    ) {
        let (sx, sy, sz) = (classify(&x.dual), classify(&y.dual), classify(&z.dual));

        let constant = sx.is_constant() & sy.is_constant() & sz.is_constant();
        let identity = sx.is_unit()
            & sy.is_unit()
            & sz.is_unit()
            & (sx.slot != sy.slot)
            & (sy.slot != sz.slot)
            & (sx.slot != sz.slot);

        if constant {
            // The harmonics do not vary either: run the inner vector's kernel on the
            // caller's table and wrap the results as constants.
            let mut values = [V::ZERO; M];
            V::spherical_harmonics_with_p::<P, L, M>(table, x.re, y.re, z.re, &mut values);

            let mut i = 0;
            while i < M {
                out[i] = Dual::constant(values[i]);
                i += 1;
            }
        } else if identity {
            // The duals are exactly d/d(x,y,z): one shared recurrence via the analytic
            // gradient form. That kernel runs in `V` (a nested dual keeps its inner
            // derivatives), so lift the primal table first (the identity copy for a
            // plain real `V`).
            let (sx, sy, sz) = (sx.slot as usize, sy.slot as usize, sz.slot as usize);
            let lifted = table.lift::<V>();

            let mut values = [V::ZERO; M];
            let mut ddx = [V::ZERO; M];
            let mut ddy = [V::ZERO; M];
            let mut ddz = [V::ZERO; M];
            sh_eval_d_impl::<V, L, M>(&lifted, x.re, y.re, z.re, &mut values, &mut ddx, &mut ddy, &mut ddz);

            let mut i = 0;
            while i < M {
                let mut dual = [V::ZERO; N];
                dual[sx] = ddx[i];
                dual[sy] = ddy[i];
                dual[sz] = ddz[i];
                out[i] = Dual::new(values[i], dual);
                i += 1;
            }
        } else {
            // A genuine Jacobian: the recurrence runs in dual arithmetic, but the
            // coefficients stay primal-typed, so each coefficient multiply is
            // `Dual * real` rather than `Dual * Dual`.
            sh_eval_mixed_impl::<Self, V, L, M>(table, x, y, z, out);
        }
    }

    /// Evaluates the basis, taking a shortcut when the direction is seeded the way
    /// callers usually seed it.
    ///
    /// * **All three inputs constant**: the harmonics do not vary either, so this runs
    ///   the inner vector's value kernel (the fully unrolled one, for a real `V`) and
    ///   wraps the results. Dual arithmetic disappears entirely.
    /// * **Identity seeding**, `x`, `y` and `z` each varying in their own slot: the
    ///   duals are then exactly `$\partial Y/\partial(x,y,z)$`, which the analytic
    ///   gradient form produces from one shared recurrence rather than by carrying three
    ///   derivatives through every operation.
    /// * **Anything else**: a genuine Jacobian, and the generic dual path is what it is
    ///   for.
    ///
    /// The classification costs `3 * N` all-lane comparisons against splat constants,
    /// against a kernel that is `O(L^2)` dual operations, so it is noise even when it
    /// declines.
    ///
    /// Note this is the _value_ form. If you want `$\partial/\partial(x,y,z)$` and
    /// nothing more, call
    /// [`spherical_harmonics_d`](thermite_special::RealPrimalMath::spherical_harmonics_d)
    /// on the real vector directly. Identity-seeding a dual to recover it works, and
    /// takes this path, but asks for a wrapper the answer never needed.
    #[inline(always)]
    fn spherical_harmonics<P: Policy, const L: usize, const M: usize, const CS: bool>(
        x: Self,
        y: Self,
        z: Self,
        out: &mut [Self; M],
    ) {
        let (sx, sy, sz) = (classify(&x.dual), classify(&y.dual), classify(&z.dual));

        // Only the constant case is handled here, because it can skip the table
        // entirely: a real `V`'s one-shot kernel is the fully-unrolled compile-time
        // form. Everything else builds the primal table once and goes through
        // `spherical_harmonics_with`, which re-classifies for the identity and
        // general-Jacobian paths (the classification is noise next to the kernels).
        if sx.is_constant() & sy.is_constant() & sz.is_constant() {
            let mut values = [V::ZERO; M];
            V::spherical_harmonics_p::<P, L, M, CS>(x.re, y.re, z.re, &mut values);

            let mut i = 0;
            while i < M {
                out[i] = Dual::constant(values[i]);
                i += 1;
            }
        } else {
            let mut table = ShTable::<<V as PrimalProjection>::Primal, M>::zeroed();
            V::spherical_harmonics_table_p::<P, L, M, CS>(&mut table);
            <Self as SpecializedRealSpecialMath<Dual<E, N>>>::spherical_harmonics_with::<P, L, M>(&table, x, y, z, out);
        }
    }

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
    fn ndtr<P: Policy>(self) -> Self {
        let x = self.re;
        let v = x.ndtr_p::<P>();
        // d/dx Phi(x) = phi(x) = e^{-x^2/2} / sqrt(2 pi)
        let factor = V::FRAC_1_SQRT_TAU * (-(x * x * V::HALF)).exp_p::<P>();
        self.chain(v, factor)
    }

    /// The derivative is the inverse Mills ratio `phi(x)/Phi(x)`, which the value kernel
    /// hands back beside the value: in the tail it is `1/(sqrt(2 pi) a)` from the `erfcx`
    /// already in hand, finite where `Phi` has underflowed, and goes to zero on the right.
    #[inline(always)]
    fn log_ndtr<P: Policy>(self) -> Self {
        let (v, mills) = <V as SpecializedRealSpecialMath<E>>::log_ndtr_with_deriv::<P>(self.re);
        self.chain(v, mills)
    }

    /// `x = inv_log_ndtr(y)` has `dx/dy = 1/(phi(x)/Phi(x))`, the implicit function
    /// theorem on the forward: the Newton loop is never differentiated.
    #[inline(always)]
    fn inv_log_ndtr<P: Policy>(self) -> Self {
        let x = self.re.inv_log_ndtr_p::<P>();
        let (_, mills) = <V as SpecializedRealSpecialMath<E>>::log_ndtr_with_deriv::<P>(x);
        self.chain(x, mills.approx_reciprocal_p::<P>())
    }

    /// `dx/dy = 1/trigamma(x)`, implicit function theorem again.
    #[inline(always)]
    fn inv_digamma<P: Policy>(self) -> Self {
        let x = self.re.inv_digamma_p::<P>();
        self.chain(x, x.trigamma_p::<P>().approx_reciprocal_p::<P>())
    }

    /// `dw/dx = w/(1 + w)` from differentiating `w + ln w = x`.
    #[inline(always)]
    fn wright_omega<P: Policy>(self) -> Self {
        let w = self.re.wright_omega_p::<P>();
        self.chain(w, w / (V::ONE + w))
    }

    /// `A' = 1 - A^2 - (2 nu - 1) A / x`, a closed form in the value. The order is treated
    /// as a constant parameter (its derivative components are ignored).
    #[inline(always)]
    fn bessel_i_ratio<P: Policy>(self, nu: Self) -> Self {
        let a = <V as SpecializedRealSpecialMath<E>>::bessel_i_ratio::<P>(self.re, nu.re);
        let da = thermite_special::specialized::bessel_i_ratio_deriv::<P, E, V>(a, self.re, nu.re);
        self.chain(a, da)
    }

    /// `d kappa / d r = 1 / A'(kappa)`, the implicit function theorem on the forward, with
    /// `A(kappa) = r` by construction so no ratio is evaluated.
    #[inline(always)]
    fn inv_bessel_i_ratio<P: Policy>(self, nu: Self) -> Self {
        let kappa = <V as SpecializedRealSpecialMath<E>>::inv_bessel_i_ratio::<P>(self.re, nu.re);
        let da = thermite_special::specialized::bessel_i_ratio_deriv::<P, E, V>(self.re, kappa, nu.re);
        self.chain(kappa, da.approx_reciprocal_p::<P>())
    }

    /// `d(1 - A)/dx = -A'`, with `A'` from the complement so nothing cancels.
    #[inline(always)]
    fn bessel_i_ratio_1m<P: Policy>(self, nu: Self) -> Self {
        let c = <V as SpecializedRealSpecialMath<E>>::bessel_i_ratio_1m::<P>(self.re, nu.re);
        let da = thermite_special::specialized::bessel_i_ratio_deriv_1m::<P, E, V>(c, self.re, nu.re);
        self.chain(c, -da)
    }

    /// `d kappa / d t = -1 / A'(kappa)`, with `1 - A(kappa) = t` by construction.
    #[inline(always)]
    fn inv_bessel_i_ratio_1m<P: Policy>(self, nu: Self) -> Self {
        let kappa = <V as SpecializedRealSpecialMath<E>>::inv_bessel_i_ratio_1m::<P>(self.re, nu.re);
        let da = thermite_special::specialized::bessel_i_ratio_deriv_1m::<P, E, V>(self.re, kappa, nu.re);
        self.chain(kappa, -da.approx_reciprocal_p::<P>())
    }

    /// Nodes and weights are constants of the rule (the index is not a variable), so the
    /// real kernel runs on the value part and the derivative components are zero.
    #[inline(always)]
    fn gauss_legendre<P: Policy>(self, n: u32) -> (Self, Self) {
        let (x, w) = self.re.gauss_legendre_p::<P>(n);
        (Self::constant(x), Self::constant(w))
    }

    #[inline(always)]
    fn gauss_hermite<P: Policy>(self, n: u32) -> (Self, Self) {
        let (x, w) = self.re.gauss_hermite_p::<P>(n);
        (Self::constant(x), Self::constant(w))
    }

    /// `alpha` is a parameter of the rule too, and its derivative components are dropped.
    #[inline(always)]
    fn gauss_laguerre<P: Policy>(self, alpha: Self, n: u32) -> (Self, Self) {
        let (x, w) = self.re.gauss_laguerre_p::<P>(alpha.re, n);
        (Self::constant(x), Self::constant(w))
    }

    /// `d/dx ln erfc(x) = -2 e^{-x^2} / (sqrt(pi) erfc(x)) = -2 / (sqrt(pi) erfcx(x))`,
    /// on the same reasoning as [`log_ndtr`](Self::log_ndtr).
    #[inline(always)]
    fn logerfc<P: Policy>(self) -> Self {
        let x = self.re;
        let v = x.logerfc_p::<P>();
        let factor = -V::FRAC_2_SQRT_PI / x.erfcx_p::<thermite_special::specialized::LogTailPolicy<P>>();
        self.chain(v, factor)
    }

    /// `(sn, cn, dn)`, differentiated **without** differentiating the Landen ladder.
    ///
    /// The triple is closed under `d/du` (`sn' = cn dn`, `cn' = -sn dn`,
    /// `dn' = -k^2 sn cn`), so once the values are in hand every derivative component is a
    /// product of values already computed. The ladder runs entirely on real vectors and
    /// three multiplies finish the job.
    ///
    /// That is better conditioned as well as faster than letting the generic body run in dual
    /// arithmetic. The ladder stops on a convergence test, `Dual` lowers comparisons
    /// to the value part alone (`Dual::cmp_le` reads `self.re`), and so a differentiated
    /// iteration stops when the _value_ has converged, with the derivative getting however
    /// many rungs that happened to buy. Taking the derivative from a closed form removes the
    /// question.
    ///
    /// The shortcut only covers a derivative in `u`. Differentiating in the **modulus** is a
    /// genuinely different computation (it needs the incomplete integral of the second kind
    /// at the amplitude, which this function never forms), so a dual-valued `k` falls back to
    /// the generic ladder in dual arithmetic and inherits the convergence caveat above. The
    /// test is on `k`'s derivative components being identically zero, which is the
    /// overwhelmingly common case: `k` is normally a fixed material or geometric parameter.
    #[inline(always)]
    fn jacobi_elliptic<P: Policy>(u: Self, k: Self) -> (Self, Self, Self) {
        let mut k_is_constant = true;
        let mut i = 0;
        while i < N {
            k_is_constant &= k.dual[i].is_zero().all();
            i += 1;
        }

        if !k_is_constant {
            return thermite_special::specialized::jacobi_elliptic_impl::<P, Dual<E, N>, Self>(u, k);
        }

        let (sn, cn, dn) = V::jacobi_elliptic_p::<P>(u.re, k.re);
        let ksq = k.re * k.re;

        (
            u.chain(sn, cn * dn),
            u.chain(cn, -(sn * dn)),
            u.chain(dn, -(ksq * sn * cn)),
        )
    }

    #[inline(always)]
    fn langevin<P: Policy>(self) -> Self {
        let x = self.re;
        let l = x.langevin_p::<P>();
        self.chain(l, langevin_deriv::<P, V>(x, l))
    }

    #[inline(always)]
    fn inv_langevin<P: Policy>(self) -> Self {
        // d/dy L^-1(y) = 1/L'(x) at x = L^-1(y).
        let x = self.re.inv_langevin_p::<P>();
        self.chain(x, langevin_deriv::<P, V>(x, self.re).approx_reciprocal_p::<P>())
    }

    // The complement pair: same derivatives up to sign.
    #[inline(always)]
    fn langevin_1m<P: Policy>(self) -> Self {
        let x = self.re;
        // `langevin_deriv` reads L on its |x| <= 1 branch, and 1 - (1 - L) has lost L
        // entirely for tiny x (the value is 1 to the last bit while L' = 1/3), so L is
        // evaluated on its own there. Only the small lanes pay the second polynomial.
        let l = if x.abs().cmp_le(V::ONE).any() {
            x.langevin_p::<P>()
        } else {
            V::ZERO
        };
        self.chain(x.langevin_1m_p::<P>(), -langevin_deriv::<P, V>(x, l))
    }

    // (`langevin_deriv` reads L only for |x| <= 1, i.e. t >= 0.69, where 1 - t is exact.)
    #[inline(always)]
    fn inv_langevin_1m<P: Policy>(self) -> Self {
        let x = self.re.inv_langevin_1m_p::<P>();
        self.chain(
            x,
            -langevin_deriv::<P, V>(x, V::ONE - self.re).approx_reciprocal_p::<P>(),
        )
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
/// never consulted on the hot path, but the real-line defaults are the right answer
/// anyway, since a dual number orders and compares by its real part.
impl<V, E: 'static, const N: usize> thermite_special::specialized::ExpIntDetails<Dual<E, N>, Dual<V, N>> for Dual<V, N> where
    Dual<V, N>: thermite::vector::FloatVector<Element = Dual<E, N>>
{
}

/// `L'(x)` given `l = L(x)`, without the real kernel's tables (they are private to
/// `thermite-special`, and `langevin_d` lives on `RealPrimalMath`, which an inner dual
/// need not have).
///
/// Below `|x| = 1`, Sra's exact identity `L' = 1 - L^2 - 2L/x` (from `L = coth x - 1/x`
/// and `coth' = 1 - coth^2`) costs no transcendental and loses at most ~2 bits. Above,
/// where `L -> 1` and the identity cancels to nothing, `1/x^2 - csch^2(x)` with
/// `csch^2 = 4q/(1-q)^2`, `q = e^{-2|x|}`, the same form the real kernel uses.
#[inline(always)]
fn langevin_deriv<P: Policy, V: DualMathVector>(x: V, l: V) -> V {
    let ax = x.abs();
    let is_small = ax.cmp_le(V::ONE);
    let rcp = ax.approx_reciprocal_p::<P>();

    // 1 - L(L + 2/x). L is odd so L/x = |L|/|x|.
    let mut dl = l.nmul_adde(l, (l.abs() + l.abs()).nmul_adde(rcp, V::ONE));
    // 0/0 at exactly zero, where L'(0) = 1/3.
    dl = x
        .is_zero()
        .select(V::splat(<V::Element as FloatElement>::ConstRatio::<1, 3>::VALUE), dl);

    if const { P::POLICY.avoid_branching } || !is_small.all() {
        // Clamped so x = inf gives 0 rather than inf*0 (see the real kernel).
        let ax = ax.min(V::MAX);
        let q = (-(ax + ax)).exp_p::<P>();
        let d = (V::ONE - q).approx_reciprocal_p::<P>();
        let csch2 = (q + q) * d * (d + d);
        dl = is_small.select(dl, rcp.mul_sube(rcp, csch2));
    }

    dl
}
