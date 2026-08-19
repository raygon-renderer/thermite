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
//! differentiation: psi_1' is psi_2, whose derivative is psi_3, and so on. Adding
//! an order to the trait moves the hole one step out instead of filling it, so the
//! ladder is cut here, one order past what the rest of the family needs.
//! Closing it properly means a general `polygamma(n)`, which *is* closed, since
//! its derivative is `polygamma(n + 1)`.
//!
//! `bessel_j`, because only order 0 exists upstream, so `J_n' = (J_{n-1} -
//! J_{n+1})/2` cannot be formed.
//!
//! Both panic if called. Everything that does not depend on them works.

use thermite::math::PrimalProjection;
use thermite::math::policy::Policy;
use thermite_special::specialized::{
    ShTable, SpecializedRealSpecialMath, SpecializedSpecialMath, sh_eval_d_impl, sh_eval_mixed_impl,
};
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
        // it, which the generic dual-arithmetic default does not.
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

    // TEMP(bessel_j): disabled until orders beyond J_0 exist. See thermite-special/src/lib.rs.
    // Would need adjacent orders J_(n-1), J_(n+1) for J_n' anyway.
    //#[inline(always)]
    //fn bessel_j<P: Policy, const M: usize>(self) -> Self {
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

// Same bound as the `SpecializedSpecialMath` impl above, which this one requires.
impl<V, E, const N: usize> SpecializedRealSpecialMath<Dual<E, N>> for Dual<V, N>
where
    V: DualSpecialVector + FloatVector<Element = E> + SpecializedSpecialMath<E>,
{
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
    fn langevin<P: Policy>(self) -> Self {
        let x = self.re;
        let l = x.langevin_p::<P>();
        self.chain(l, langevin_deriv::<P, V>(x, l))
    }

    #[inline(always)]
    fn inv_langevin<P: Policy>(self) -> Self {
        // d/dy L^-1(y) = 1/L'(x) at x = L^-1(y).
        let x = self.re.inv_langevin_p::<P>();
        self.chain(x, langevin_deriv::<P, V>(x, self.re).reciprocal_p::<P>())
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
        self.chain(x, -langevin_deriv::<P, V>(x, V::ONE - self.re).reciprocal_p::<P>())
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
    let rcp = ax.reciprocal_p::<P>();

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
        let d = (V::ONE - q).reciprocal_p::<P>();
        let csch2 = (q + q) * d * (d + d);
        dl = is_small.select(dl, rcp.mul_sube(rcp, csch2));
    }

    dl
}
